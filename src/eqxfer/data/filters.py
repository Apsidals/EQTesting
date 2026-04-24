"""Feature cache. Compute physics / site features once per (loader_config_hash)
and parquet them to data/processed/. Every rung reuses the cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import LoaderConfig, REGION_BBOXES, config_hash
from ..features.pd_tauc import (
    compute_corner_frequency,
    compute_pd,
    compute_pgv,
    compute_tau_c,
    infer_instrument_kind,
)
from .geological import compute_site_features_table
from .stead_loader import SteadLoader

CACHE_DIR = Path("data/processed")

PHYSICS_FEATURE_COLUMNS = ("pd_z", "tau_c", "pgv_z", "fc_z")


def _cache_path(cfg: LoaderConfig) -> Path:
    return CACHE_DIR / f"features_{config_hash(cfg)}.parquet"


def _site_cache_path(cfg: LoaderConfig) -> Path:
    return CACHE_DIR / f"site_{config_hash(cfg)}.parquet"


def load_or_compute_features(loader: SteadLoader) -> pd.DataFrame:
    """Return a DataFrame indexed by trace_name with physics features.

    Columns: pd_z (rung 1), tau_c / pgv_z / fc_z (probe targets for rung 5).
    If an existing cache is missing any required columns, it is rebuilt.
    """
    cache_path = _cache_path(loader.config)
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        missing = [c for c in PHYSICS_FEATURE_COLUMNS if c not in cached.columns]
        if not missing:
            print(f"Loading cached features: {cache_path}", flush=True)
            return cached
        print(
            f"Cached features missing columns {missing}; recomputing to {cache_path}",
            flush=True,
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    receiver_types = dict(
        zip(loader.metadata["trace_name"], loader.metadata["receiver_type"])
    )

    rows: list[dict] = []
    sr = loader.config.sample_rate_hz
    for name, window in tqdm(
        loader.iter_waveforms(), total=len(loader), desc="phys-features", unit="trace"
    ):
        kind = infer_instrument_kind(receiver_types[name])
        rows.append(
            {
                "trace_name": name,
                "pd_z": compute_pd(window, sr, kind),
                "tau_c": compute_tau_c(window, sr, kind),
                "pgv_z": compute_pgv(window, sr, kind),
                "fc_z": compute_corner_frequency(window, sr, kind),
            }
        )

    df = pd.DataFrame(rows).set_index("trace_name")
    df.to_parquet(cache_path)
    print(f"  cached {len(df):,} rows → {cache_path}", flush=True)
    return df


def load_or_compute_site_features(
    loader: SteadLoader,
    vs30_path: Path = Path("data/raw/vs30/global_vs30.grd"),
    crust1_path: Path = Path("data/raw/crust1/CRUST1.0-vp.r0.1.nc"),
) -> pd.DataFrame:
    """Return per-trace site features (Vs30, crustal Vp, sediment thickness,
    NEHRP one-hot, instrument one-hot). Cached per loader-config hash."""
    cache_path = _site_cache_path(loader.config)
    if cache_path.exists():
        print(f"Loading cached site features: {cache_path}", flush=True)
        return pd.read_parquet(cache_path)

    print(f"Computing site features (no cache at {cache_path})...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bbox = REGION_BBOXES[loader.config.region]
    df = compute_site_features_table(
        metadata=loader.metadata,
        vs30_path=vs30_path,
        crust1_path=crust1_path,
        bbox=bbox,
    )
    df.to_parquet(cache_path)
    print(f"  cached {len(df):,} rows → {cache_path}", flush=True)
    return df
