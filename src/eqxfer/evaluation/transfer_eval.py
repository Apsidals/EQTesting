"""Transfer protocol orchestration.

Three things a transfer experiment has to do:
1. Zero-shot: run the California-pretrained SplitTransferModel on target
   data, no parameter changes.
2. Few-shot: freeze the universal encoder, re-fit site encoder + fusion
   on n_events target events (n_events ∈ {100, 500, 2000}). Compare
   against a from-scratch baseline on the same budget.
3. Efficiency crossover: at what n_events does a from-scratch model match
   the zero-shot transfer? If ≲ 500, the universal encoder is carrying
   real physics knowledge across tectonic regimes.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import ModelConfig, TrainConfig, TransferConfig
from ..data.dataset import WaveformDataset, make_stratified_sampler
from ..data.splits import few_shot_event_sample
from ..evaluation.metrics import MetricPanel, compute_metric_panel, format_panel
from ..models.split_transfer import SplitTransferModel
from ..training.loops import TrainResult, predict, train_model


@dataclass
class TransferStageResult:
    stage: str       # "zero_shot", "few_shot_n=100", "from_scratch_n=100", ...
    n_adapt_events: int
    panel: MetricPanel
    train_result: TrainResult | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_adapt_events": self.n_adapt_events,
            "panel": self.panel.to_dict(),
            "train_result": (self.train_result.to_dict() if self.train_result else None),
        }


@dataclass
class TransferReport:
    target_region: str
    stages: list[TransferStageResult] = field(default_factory=list)
    mmd_source_target: float | None = None
    silhouette_source_target: float | None = None
    probe_r2: dict[str, float] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        payload = {
            "target_region": self.target_region,
            "stages": [s.to_dict() for s in self.stages],
            "mmd_source_target": self.mmd_source_target,
            "silhouette_source_target": self.silhouette_source_target,
            "probe_r2": self.probe_r2,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))


def build_loader(
    dataset: WaveformDataset,
    batch_size: int,
    num_workers: int,
    sampler: torch.utils.data.Sampler | None = None,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def zero_shot_evaluate(
    model: SplitTransferModel,
    target_test_loader: DataLoader,
    device: torch.device,
) -> MetricPanel:
    y_true, y_pred = predict(model, target_test_loader, device)
    return compute_metric_panel(y_true, y_pred)


def few_shot_finetune(
    pretrained_state: dict,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    transfer_cfg: TransferConfig,
    target_metadata: pd.DataFrame,
    target_site_features: pd.DataFrame,
    loader_cfg,
    n_events: int,
    val_trace_names: list[str],
    test_loader: DataLoader,
    device: torch.device,
    exp_dir: Path,
) -> TransferStageResult:
    """Sample n_events target events, fine-tune site+fusion (universal
    encoder frozen by default), eval on test."""
    adapt_trace_names = few_shot_event_sample(
        metadata=target_metadata,
        n_events=n_events,
        seed=transfer_cfg.seed,
    )
    # Drop any traces that would leak into val (should already be disjoint
    # since few_shot samples from training-portion events, but belt+braces).
    val_set = set(val_trace_names)
    adapt_trace_names = [t for t in adapt_trace_names if t not in val_set]

    adapt_ds = WaveformDataset(
        loader_cfg=loader_cfg,
        metadata=target_metadata,
        site_features=target_site_features,
        trace_names=adapt_trace_names,
    )
    sampler = make_stratified_sampler(
        adapt_ds.magnitudes_array(),
        bin_width=train_cfg.bin_width,
        cap=train_cfg.sampler_cap,
        seed=train_cfg.seed,
    )
    val_ds = WaveformDataset(
        loader_cfg=loader_cfg,
        metadata=target_metadata,
        site_features=target_site_features,
        trace_names=val_trace_names,
    )
    train_loader = build_loader(
        adapt_ds,
        batch_size=min(train_cfg.batch_size, len(adapt_ds)),
        num_workers=train_cfg.num_workers,
        sampler=sampler,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
    )

    model = SplitTransferModel(model_cfg)
    model.load_state_dict(pretrained_state)
    if transfer_cfg.freeze_universal:
        model.freeze_universal()
    model.to(device)

    fs_train_cfg = train_cfg.model_copy(
        update={"epochs": transfer_cfg.few_shot_epochs, "lr": transfer_cfg.few_shot_lr}
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=fs_train_cfg,
        device=device,
        exp_dir=exp_dir,
        trainable_params=trainable,
        tag=f"few_shot_n{n_events}",
    )

    y_true, y_pred = predict(model, test_loader, device)
    panel = compute_metric_panel(y_true, y_pred)
    print(f"\n--- Few-shot n={n_events} test panel ---\n{format_panel(panel)}")
    return TransferStageResult(
        stage=f"few_shot_n{n_events}",
        n_adapt_events=n_events,
        panel=panel,
        train_result=result,
    )


def from_scratch_baseline(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    transfer_cfg: TransferConfig,
    target_metadata: pd.DataFrame,
    target_site_features: pd.DataFrame,
    loader_cfg,
    n_events: int,
    val_trace_names: list[str],
    test_loader: DataLoader,
    device: torch.device,
    exp_dir: Path,
) -> TransferStageResult:
    """Train the full split architecture on `n_events` target events WITHOUT
    pretraining. This is the data-efficiency baseline that transfer must
    beat at low budgets for the hypothesis to hold."""
    adapt_trace_names = few_shot_event_sample(
        metadata=target_metadata,
        n_events=n_events,
        seed=transfer_cfg.seed + 1,  # different sample than few-shot
    )
    val_set = set(val_trace_names)
    adapt_trace_names = [t for t in adapt_trace_names if t not in val_set]

    adapt_ds = WaveformDataset(
        loader_cfg=loader_cfg,
        metadata=target_metadata,
        site_features=target_site_features,
        trace_names=adapt_trace_names,
    )
    val_ds = WaveformDataset(
        loader_cfg=loader_cfg,
        metadata=target_metadata,
        site_features=target_site_features,
        trace_names=val_trace_names,
    )
    sampler = make_stratified_sampler(
        adapt_ds.magnitudes_array(),
        bin_width=train_cfg.bin_width,
        cap=train_cfg.sampler_cap,
        seed=train_cfg.seed,
    )
    train_loader = build_loader(
        adapt_ds,
        batch_size=min(train_cfg.batch_size, len(adapt_ds)),
        num_workers=train_cfg.num_workers,
        sampler=sampler,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
    )

    model = SplitTransferModel(model_cfg)
    fs_train_cfg = train_cfg.model_copy(
        update={"epochs": transfer_cfg.from_scratch_epochs}
    )
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=fs_train_cfg,
        device=device,
        exp_dir=exp_dir,
        tag=f"from_scratch_n{n_events}",
    )

    y_true, y_pred = predict(model, test_loader, device)
    panel = compute_metric_panel(y_true, y_pred)
    print(
        f"\n--- From-scratch n={n_events} test panel ---\n{format_panel(panel)}"
    )
    return TransferStageResult(
        stage=f"from_scratch_n{n_events}",
        n_adapt_events=n_events,
        panel=panel,
        train_result=result,
    )


def clone_state(state: dict) -> dict:
    return copy.deepcopy({k: v.detach().clone() for k, v in state.items()})
