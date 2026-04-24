"""Supervised training loop for the split architecture.

GPU is the default: CLAUDE.md requires GPU training. If `require_cuda=True`
and CUDA isn't available, we raise rather than silently CPU-train.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..evaluation.metrics import MetricPanel, compute_metric_panel, format_panel_compact
from .losses import huber_loss
from .schedulers import cosine_warmup_scheduler


def select_device(require_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] CUDA: {name} ({total_gb:.1f} GB)", flush=True)
        return dev
    if require_cuda:
        raise RuntimeError(
            "CUDA is required by TrainConfig.require_cuda but torch.cuda.is_available() "
            "is False. Install a CUDA-enabled torch wheel, or set require_cuda=False "
            "in the run config to permit CPU training (not recommended for rung 5+)."
        )
    print("[device] CUDA not available — falling back to CPU (slow!).", flush=True)
    return torch.device("cpu")


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_rmse: float
    val_r2: float
    val_bias: float
    val_acc_0p3: float
    val_mae_m3: float
    val_mae_m4: float
    val_mae_m5: float
    val_mae_m6plus: float
    val_n_m6plus: int
    lr: float
    seconds: float


@dataclass
class TrainResult:
    best_val_mae: float
    best_epoch: int
    best_state_path: Path
    history: list[EpochRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_val_mae": self.best_val_mae,
            "best_epoch": self.best_epoch,
            "best_state_path": str(self.best_state_path),
            "history": [vars(h) for h in self.history],
        }


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> tuple[float, MetricPanel]:
    """Run `model` over `loader` once. Returns (avg_loss, full MetricPanel).

    Bootstrap CIs are skipped here so this can run every epoch without
    noticeable overhead — CIs are only meaningful on the final test panel."""
    model.eval()
    total_loss = 0.0
    n = 0
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for waveform, site_feats, y in loader:
            waveform = waveform.to(device, non_blocking=True)
            site_feats = site_feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda", enabled=amp_enabled, dtype=torch.bfloat16
            ):
                y_hat = model(waveform, site_feats).float()
                loss = loss_fn(y_hat, y)
            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            n += bs
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(y_hat.detach().cpu().numpy())
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return float("nan"), compute_metric_panel(empty, empty, bootstrap_samples=0)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    panel = compute_metric_panel(y_true, y_pred, bootstrap_samples=0)
    return total_loss / n, panel


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    exp_dir: Path,
    trainable_params: list[torch.nn.Parameter] | None = None,
    tag: str = "model",
) -> TrainResult:
    """Train `model`. If `trainable_params` is given (few-shot mode), only
    those params get gradient updates and go into the optimizer.

    Writes `{tag}_best.pt` and `{tag}_history.json` into exp_dir.
    """
    model.to(device)
    loss_fn = huber_loss(delta=cfg.huber_delta).to(device)

    params = (
        trainable_params
        if trainable_params is not None
        else [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
    )
    # AMP uses bfloat16, NOT float16. The raw STEAD waveform amplitudes
    # aren't normalized (hard rule — magnitude IS amplitude), so M5+ events
    # push FP16's ~65k max straight into overflow → NaN. bf16 has FP32's
    # exponent range, no overflow, and no GradScaler needed.
    amp_enabled = cfg.amp and device.type == "cuda"

    best_val_mae = float("inf")
    best_epoch = -1
    best_state_path = exp_dir / f"{tag}_best.pt"
    best_state: dict | None = None
    epochs_since_best = 0
    history: list[EpochRecord] = []

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for waveform, site_feats, y in train_loader:
            waveform = waveform.to(device, non_blocking=True)
            site_feats = site_feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=amp_enabled, dtype=torch.bfloat16
            ):
                y_hat = model(waveform, site_feats)
                loss = loss_fn(y_hat, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"loss is non-finite at epoch {epoch} — check for degenerate "
                    f"inputs. y_hat range=[{y_hat.min().item()}, {y_hat.max().item()}], "
                    f"y range=[{y.min().item()}, {y.max().item()}]"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()

            bs = y.shape[0]
            running += float(loss.item()) * bs
            n += bs

        scheduler.step()
        train_loss = running / max(1, n)
        val_loss, val_panel = _evaluate(model, val_loader, device, loss_fn)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        val_mae = val_panel.mae
        rec = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mae=val_mae,
            val_rmse=val_panel.rmse,
            val_r2=val_panel.r2,
            val_bias=val_panel.bias,
            val_acc_0p3=val_panel.acc_0p3,
            val_mae_m3=val_panel.mae_m3,
            val_mae_m4=val_panel.mae_m4,
            val_mae_m5=val_panel.mae_m5,
            val_mae_m6plus=val_panel.mae_m6plus,
            val_n_m6plus=val_panel.n_m6plus,
            lr=lr,
            seconds=elapsed,
        )
        history.append(rec)
        print(
            f"[{tag}] epoch {epoch+1:3d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={lr:.2e}  ({elapsed:.1f}s)\n"
            f"         {format_panel_compact(val_panel)}",
            flush=True,
        )

        if val_mae < best_val_mae - 1e-5:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
            torch.save(best_state, best_state_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.early_stopping_patience:
                print(
                    f"[{tag}] early stopping at epoch {epoch+1} "
                    f"(no val improvement for {epochs_since_best} epochs)",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    (exp_dir / f"{tag}_history.json").write_text(
        json.dumps([vars(h) for h in history], indent=2)
    )
    return TrainResult(
        best_val_mae=best_val_mae,
        best_epoch=best_epoch,
        best_state_path=best_state_path,
        history=history,
    )


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full forward pass over `loader`. Returns (y_true, y_pred) as 1D arrays."""
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    with torch.no_grad():
        for waveform, site_feats, y in loader:
            waveform = waveform.to(device, non_blocking=True)
            site_feats = site_feats.to(device, non_blocking=True)
            y_hat = model(waveform, site_feats)
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(y_hat.detach().cpu().numpy())
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def extract_physics_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return (N, phys_embed_dim) embeddings — frozen outputs of the universal
    encoder. Used by probes and alignment metrics."""
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for waveform, _, _ in loader:
            waveform = waveform.to(device, non_blocking=True)
            emb = model.encode_physics(waveform)
            out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0)
