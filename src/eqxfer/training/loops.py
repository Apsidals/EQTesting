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

from ..config import AuxLossConfig, SeparationLossConfig, TrainConfig
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
        for batch in loader:
            waveform = batch[0].to(device, non_blocking=True)
            site_feats = batch[1].to(device, non_blocking=True)
            y = batch[2].to(device, non_blocking=True)
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


def _stratified_val_mae(panel: MetricPanel) -> float:
    """Macro-average MAE over populated magnitude bins. Used for early
    stopping instead of aggregate MAE — aggregate MAE is dominated by
    M[3-4] which is 95%+ of California val, so "best val MAE" otherwise
    selects for predicting the mean well and ignoring large events."""
    bins = [
        (panel.mae_m3, panel.n_m3),
        (panel.mae_m4, panel.n_m4),
        (panel.mae_m5, panel.n_m5),
        (panel.mae_m6plus, panel.n_m6plus),
    ]
    values = [mae for mae, n in bins if n > 0 and np.isfinite(mae)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    exp_dir: Path,
    trainable_params: list[torch.nn.Parameter] | None = None,
    tag: str = "model",
    aux_cfg: AuxLossConfig | None = None,
    sep_cfg: SeparationLossConfig | None = None,
) -> TrainResult:
    """Train `model`. If `trainable_params` is given (few-shot mode), only
    those params get gradient updates and go into the optimizer.

    When `aux_cfg` is provided and the model exposes `forward_with_aux`,
    auxiliary physics-regression losses (log10 fc / tau_c / pd) are added
    to the magnitude loss.

    When `sep_cfg` is provided, the train_loader MUST yield batches
    structured as (events_per_batch × stations_per_event) contiguous blocks
    (i.e. use EventGroupedBatchSampler). The loop reshapes the physics
    embedding to (events, stations, D) and adds L_sep = mean within-event
    variance to the total loss.

    Few-shot and from-scratch baselines pass None for both.

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

    use_aux = aux_cfg is not None and hasattr(model, "forward_with_aux")
    use_sep = sep_cfg is not None and hasattr(model, "forward_with_aux")
    need_phys = use_aux or use_sep
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        running_mag = 0.0
        running_aux = 0.0
        running_sep = 0.0
        n = 0
        for batch in train_loader:
            waveform = batch[0].to(device, non_blocking=True)
            site_feats = batch[1].to(device, non_blocking=True)
            y = batch[2].to(device, non_blocking=True)

            if use_aux and len(batch) >= 6:
                aux_log_fc = batch[3].to(device, non_blocking=True)
                aux_log_tau_c = batch[4].to(device, non_blocking=True)
                aux_log_pd = batch[5].to(device, non_blocking=True)
            else:
                aux_log_fc = aux_log_tau_c = aux_log_pd = None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=amp_enabled, dtype=torch.bfloat16
            ):
                if need_phys:
                    y_hat, a_fc, a_tc, a_pd, phys = model.forward_with_aux(
                        waveform, site_feats
                    )
                else:
                    y_hat = model(waveform, site_feats)
                    phys = None

                mag_loss = loss_fn(y_hat, y)

                if use_aux and aux_log_fc is not None:
                    # MSE on log-space targets, not Huber: targets are
                    # well-behaved and we want a strong gradient near zero.
                    aux_fc_loss = torch.nn.functional.mse_loss(a_fc, aux_log_fc)
                    aux_tc_loss = torch.nn.functional.mse_loss(a_tc, aux_log_tau_c)
                    aux_pd_loss = torch.nn.functional.mse_loss(a_pd, aux_log_pd)
                    aux_total = (
                        aux_cfg.fc_weight * aux_fc_loss
                        + aux_cfg.tau_c_weight * aux_tc_loss
                        + aux_cfg.pd_weight * aux_pd_loss
                    )
                else:
                    aux_total = torch.zeros((), device=device)

                if use_sep and phys is not None:
                    # Batch layout contract: EventGroupedBatchSampler packs
                    # stations_per_event contiguous traces per event. Reshape
                    # and compute within-event variance of the physics
                    # embedding — the paper's "same source → same physics
                    # embedding" claim, turned into a loss.
                    k = sep_cfg.stations_per_event
                    e = sep_cfg.events_per_batch
                    if phys.shape[0] != e * k:
                        raise RuntimeError(
                            f"batch size {phys.shape[0]} != events_per_batch "
                            f"({e}) * stations_per_event ({k}) = {e*k}; "
                            f"L_sep requires EventGroupedBatchSampler."
                        )
                    phys_grouped = phys.view(e, k, -1)
                    # unbiased=False (population variance, not sample): with
                    # small k the Bessel correction adds noise.
                    sep_loss = phys_grouped.var(dim=1, unbiased=False).mean()
                else:
                    sep_loss = torch.zeros((), device=device)

                sep_total = (
                    sep_cfg.weight * sep_loss
                    if use_sep
                    else torch.zeros((), device=device)
                )
                loss = mag_loss + aux_total + sep_total

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
            running_mag += float(mag_loss.item()) * bs
            running_aux += float(aux_total.item()) * bs
            running_sep += float(sep_total.item()) * bs
            n += bs

        scheduler.step()
        train_loss = running / max(1, n)
        train_mag_loss = running_mag / max(1, n)
        train_aux_loss = running_aux / max(1, n)
        train_sep_loss = running_sep / max(1, n)
        val_loss, val_panel = _evaluate(model, val_loader, device, loss_fn)
        val_strat_mae = _stratified_val_mae(val_panel)
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
        suffix_parts = []
        if use_aux:
            suffix_parts.append(f"mag={train_mag_loss:.4f} aux={train_aux_loss:.4f}")
        if use_sep:
            suffix_parts.append(f"sep={train_sep_loss:.4f}")
        suffix = f" ({' '.join(suffix_parts)})" if suffix_parts else ""
        print(
            f"[{tag}] epoch {epoch+1:3d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}{suffix}  "
            f"val_loss={val_loss:.4f}  val_strat_mae={val_strat_mae:.4f}  "
            f"lr={lr:.2e}  ({elapsed:.1f}s)\n"
            f"         {format_panel_compact(val_panel)}",
            flush=True,
        )

        # Early stopping on stratified val MAE, not aggregate. Aggregate is
        # dominated by M[3-4] and otherwise selects for mean-prediction.
        if np.isfinite(val_strat_mae) and val_strat_mae < best_val_mae - 1e-5:
            best_val_mae = val_strat_mae
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
        for batch in loader:
            waveform = batch[0].to(device, non_blocking=True)
            site_feats = batch[1].to(device, non_blocking=True)
            y = batch[2]
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
        for batch in loader:
            waveform = batch[0].to(device, non_blocking=True)
            emb = model.encode_physics(waveform)
            out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0)
