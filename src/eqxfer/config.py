"""Pydantic config models + config hashing. Hash is the cache key for
preprocessed features, so any field that affects the output of the
loader/feature pipeline must live in one of these models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


RegionName = Literal["California", "Greece", "Chile"]


REGION_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max)
    "California": (32.5, 42.0, -124.5, -114.0),
    "Greece":     (34.8, 41.8,  19.0,   28.5),
    "Chile":      (-56.0, -17.0, -76.0, -66.0),
}

# Legacy alias kept for callers that imported it pre-multi-region.
CALIFORNIA_BBOX = REGION_BBOXES["California"]


class LoaderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stead_dir: Path = Path("data/raw/stead")
    csv_name: str = "merge.csv"
    hdf5_name: str = "merge.hdf5"
    region: RegionName = "California"
    magnitude_min: float = 3.0
    magnitude_scales: tuple[str, ...] = ("ml", "mw")
    snr_min_db: float = 10.0
    window_samples: int = 500
    sample_rate_hz: int = 100
    bandpass_low_hz: float = 0.075
    bandpass_high_hz: float = 25.0
    bandpass_order: int = 4

    @field_validator("magnitude_scales", mode="before")
    @classmethod
    def _normalize_scales(cls, v: object) -> tuple[str, ...]:
        """Accept str or list[str]; always return a sorted lowercase tuple
        so the config hash is invariant under scale ordering."""
        if isinstance(v, str):
            return (v.lower(),)
        if isinstance(v, (list, tuple)):
            return tuple(sorted(s.lower() for s in v))
        raise ValueError(f"magnitude_scales must be a string or list of strings, got {type(v)}")


class SplitConfig(BaseModel):
    """Split strategy config.

    california_ridgecrest — geographic holdout box (test) + event-ID-grouped
        train/val over the rest of California.
    event_grouped         — no geographic holdout; event-ID-grouped
        train/val/test. Used for Greece/Chile where data is too thin for a
        geographic box. Still blocks event-level leakage, which is the
        main hard rule.
    """

    model_config = ConfigDict(frozen=True)

    strategy: Literal["california_ridgecrest", "event_grouped"] = "california_ridgecrest"
    test_lat_min: float = 35.3
    test_lat_max: float = 36.0
    test_lon_min: float = -118.0
    test_lon_max: float = -117.0
    val_fraction: float = 0.10
    test_fraction: float = 0.10  # only used when strategy == event_grouped
    seed: int = 0


class ModelConfig(BaseModel):
    """Hyperparameters for the split architecture (rung 5).

    Physics-branch channel counts kept modest — v1's over-parameterization is
    a recurring failure mode called out in CLAUDE.md.
    """

    model_config = ConfigDict(frozen=True)

    # Universal physics encoder (1D CNN on 3-component waveform).
    phys_channels: tuple[int, ...] = (32, 64, 128, 128)
    phys_kernel_size: int = 7
    phys_stride: int = 2
    phys_groupnorm_groups: int = 8
    phys_embed_dim: int = 128
    phys_dropout: float = 0.1

    # Regional site encoder (MLP on site features).
    site_hidden: tuple[int, ...] = (64, 64)
    site_embed_dim: int = 32
    site_dropout: float = 0.1

    # Fusion head (concat → MLP → scalar magnitude).
    fusion_hidden: tuple[int, ...] = (128, 64)
    fusion_dropout: float = 0.1


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    epochs: int = 60
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 10
    bin_width: float = 1.0
    sampler_cap: float = 20.0  # per hard rule #8
    require_cuda: bool = True  # fail loudly rather than silently CPU-train
    seed: int = 0


class TransferConfig(BaseModel):
    """Transfer-learning protocol for a single target region.

    Zero-shot eval uses the source-pretrained model unchanged. Few-shot
    re-fits only the site encoder + fusion head on n target events (universal
    encoder frozen). The from-scratch baseline trains the full split
    architecture on the same n target events without pretraining."""

    model_config = ConfigDict(frozen=True)

    target_region: RegionName
    few_shot_n: tuple[int, ...] = (100, 500, 2000)
    few_shot_epochs: int = 30
    few_shot_lr: float = 5e-4
    from_scratch_epochs: int = 60
    freeze_universal: bool = True
    seed: int = 0


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rung: int
    model_name: str
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model_hparams: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    transfer: TransferConfig | None = None
    notes: str = ""


def config_hash(cfg: BaseModel) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_run_config(path: Path) -> RunConfig:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return RunConfig(**raw)
