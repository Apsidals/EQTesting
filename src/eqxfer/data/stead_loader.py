"""STEAD loader + filters. Reads merge.csv and merge.hdf5, applies the
CLAUDE.md "Data processing pipeline" filters, and exposes preprocessed
(3, 500) ZNE windows via lazy HDF5 access."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from ..config import REGION_BBOXES, LoaderConfig
from ..features.waveform import preprocess


def _parse_snr(value: object) -> list[float]:
    """STEAD snr_db is a string like '[11.2 9.5 14.1]' (one per component)."""
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    if not isinstance(value, str):
        raise ValueError(f"unparseable snr_db: {value!r}")
    cleaned = value.strip("[]").replace(",", " ").split()
    return [float(v) for v in cleaned if v]


def _apply_filters(df: pd.DataFrame, cfg: LoaderConfig) -> pd.DataFrame:
    # Earthquake only.
    df = df[df["trace_category"] == "earthquake_local"].copy()

    # Regional bounding box (selected by cfg.region).
    lat_min, lat_max, lon_min, lon_max = REGION_BBOXES[cfg.region]
    df = df[
        df["source_latitude"].between(lat_min, lat_max)
        & df["source_longitude"].between(lon_min, lon_max)
    ].copy()

    # Magnitude scale(s).
    df = df[df["source_magnitude_type"].str.lower().isin(cfg.magnitude_scales)].copy()

    # Magnitude floor.
    df = df[df["source_magnitude"] >= cfg.magnitude_min].copy()

    # SNR: minimum across the 3 components >= threshold.
    snr_values = df["snr_db"].apply(_parse_snr)
    snr_min_per_trace = snr_values.apply(lambda xs: min(xs) if xs else np.nan)
    df = df[snr_min_per_trace >= cfg.snr_min_db].copy()

    # Window availability: P arrival + 500 samples must fit inside the trace.
    # STEAD traces are 6000 samples long at 100 Hz.
    required_end = df["p_arrival_sample"].astype(int) + cfg.window_samples
    df = df[required_end <= 6000].copy()

    # Must have a usable receiver_type (2-char code prefix).
    df = df[df["receiver_type"].str.len() >= 2].copy()

    df = df.reset_index(drop=True)
    return df


class SteadLoader:
    """Lazy loader: metadata in memory, waveforms on demand from HDF5."""

    def __init__(self, config: LoaderConfig) -> None:
        self.config = config
        csv_path = Path(config.stead_dir) / config.csv_name
        hdf5_path = Path(config.stead_dir) / config.hdf5_name
        if not csv_path.exists():
            raise FileNotFoundError(f"STEAD metadata not found: {csv_path}")
        if not hdf5_path.exists():
            raise FileNotFoundError(f"STEAD waveforms not found: {hdf5_path}")
        self._hdf5_path = hdf5_path

        print(f"Reading {csv_path}...", flush=True)
        full = pd.read_csv(csv_path, low_memory=False)
        print(f"  loaded {len(full):,} rows", flush=True)
        self.metadata = _apply_filters(full, config)
        print(f"  after filters: {len(self.metadata):,} traces", flush=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def trace_names(self) -> list[str]:
        return self.metadata["trace_name"].tolist()

    def get_waveform(self, trace_name: str) -> np.ndarray:
        """Return preprocessed (3, 500) ZNE window for a single trace."""
        row = self.metadata.loc[self.metadata["trace_name"] == trace_name]
        if len(row) != 1:
            raise KeyError(f"trace_name {trace_name!r} not in filtered metadata")
        p_sample = int(row["p_arrival_sample"].iloc[0])
        with h5py.File(self._hdf5_path, "r") as f:
            raw = f["data"][trace_name][:]
        return preprocess(
            raw_trace_stead=np.asarray(raw, dtype=np.float32),
            p_sample=p_sample,
            sample_rate_hz=self.config.sample_rate_hz,
            window_samples=self.config.window_samples,
            bandpass_low_hz=self.config.bandpass_low_hz,
            bandpass_high_hz=self.config.bandpass_high_hz,
            bandpass_order=self.config.bandpass_order,
        )

    def iter_waveforms(
        self, trace_names: Iterable[str] | None = None
    ) -> Iterable[tuple[str, np.ndarray]]:
        """Stream preprocessed windows from HDF5. Opens file once."""
        names = list(trace_names) if trace_names is not None else self.trace_names()
        p_samples = dict(
            zip(
                self.metadata["trace_name"],
                self.metadata["p_arrival_sample"].astype(int),
            )
        )
        with h5py.File(self._hdf5_path, "r") as f:
            data = f["data"]
            for name in names:
                raw = np.asarray(data[name][:], dtype=np.float32)
                yield name, preprocess(
                    raw_trace_stead=raw,
                    p_sample=p_samples[name],
                    sample_rate_hz=self.config.sample_rate_hz,
                    window_samples=self.config.window_samples,
                    bandpass_low_hz=self.config.bandpass_low_hz,
                    bandpass_high_hz=self.config.bandpass_high_hz,
                    bandpass_order=self.config.bandpass_order,
                )
