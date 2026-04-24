"""PyTorch Dataset wrapping SteadLoader for deep-model training.

Design choices:
- HDF5 file handle opened lazily per-worker. h5py is not fork-safe; opening
  on first __getitem__ means each DataLoader worker gets its own handle.
- Waveforms are preprocessed on the fly. The 44 GB raw HDF5 can't fit in
  RAM but per-trace decode is ~1 ms, so with num_workers>=4 we keep the
  GPU fed.
- Returns float32 tensors to match the model; amplitude is NOT normalized
  (per hard rule — magnitude is amplitude).
- Site features are precomputed and indexed by trace_name, passed in at
  construction.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..config import LoaderConfig
from ..features.waveform import preprocess
from ..models.pd_linear import bin_balanced_weights
from .waveform_cache import WaveformCache


class WaveformDataset(Dataset):
    """Returns (waveform, site_feats, magnitude, aux_log_fc, aux_log_tau_c,
    aux_log_pd) for one trace.

    When phys_features is None, the three aux slots are zeros and the
    training loop ignores them (aux_cfg=None path). This keeps the eval /
    transfer call sites — which don't need aux targets — simple: they pass
    no phys_features and unpack only the first three elements of the batch.

    waveform:   float32 tensor of shape (3, window_samples), ZNE order
    site_feats: float32 tensor of shape (SITE_FEATURE_DIM,)
    magnitude:  float32 tensor, scalar
    aux_log_*:  float32 tensor, scalar — log10 of physics feature, or 0.0
    """

    def __init__(
        self,
        loader_cfg: LoaderConfig,
        metadata: pd.DataFrame,
        site_features: pd.DataFrame,
        trace_names: list[str],
        phys_features: pd.DataFrame | None = None,
        waveform_cache: WaveformCache | None = None,
    ) -> None:
        self.loader_cfg = loader_cfg
        self.trace_names = list(trace_names)

        idx = metadata.set_index("trace_name").loc[self.trace_names]
        self.p_samples = idx["p_arrival_sample"].astype(int).to_numpy()
        self.magnitudes = idx["source_magnitude"].astype(np.float32).to_numpy()

        feat = site_features.loc[self.trace_names]
        self.site_feats = feat.to_numpy(dtype=np.float32)

        if phys_features is not None:
            pf = phys_features.loc[self.trace_names]
            # Fail loud (hard rule: no silent defaults). log10(0) / log10(NaN)
            # would silently poison aux loss and drag the encoder sideways.
            for col in ("fc_z", "tau_c", "pd_z"):
                vals = pf[col].to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
                    bad = int((~np.isfinite(vals)).sum() + (vals <= 0.0).sum())
                    raise ValueError(
                        f"phys_features['{col}'] contains {bad} non-positive "
                        f"or non-finite values; cannot log-transform for aux loss"
                    )
            self.aux_log_fc = np.log10(
                pf["fc_z"].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            self.aux_log_tau_c = np.log10(
                pf["tau_c"].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            self.aux_log_pd = np.log10(
                pf["pd_z"].to_numpy(dtype=np.float64)
            ).astype(np.float32)
            self._has_aux = True
        else:
            n = len(self.trace_names)
            self.aux_log_fc = np.zeros(n, dtype=np.float32)
            self.aux_log_tau_c = np.zeros(n, dtype=np.float32)
            self.aux_log_pd = np.zeros(n, dtype=np.float32)
            self._has_aux = False

        self._hdf5_path = Path(loader_cfg.stead_dir) / loader_cfg.hdf5_name
        self._hdf5: h5py.File | None = None  # opened lazily per worker
        self._cache = waveform_cache  # None → fall back to HDF5 + preprocess

    def __len__(self) -> int:
        return len(self.trace_names)

    def _file(self) -> h5py.File:
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self._hdf5_path, "r")
        return self._hdf5

    def __getitem__(
        self, i: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        name = self.trace_names[i]
        if self._cache is not None:
            # Per-worker lazy open: memmap + index dict are rebuilt on first
            # access in each DataLoader worker process.
            self._cache.open()
            window = self._cache.get(name)
        else:
            f = self._file()
            raw = np.asarray(f["data"][name][:], dtype=np.float32)
            window = preprocess(
                raw_trace_stead=raw,
                p_sample=int(self.p_samples[i]),
                sample_rate_hz=self.loader_cfg.sample_rate_hz,
                window_samples=self.loader_cfg.window_samples,
                bandpass_low_hz=self.loader_cfg.bandpass_low_hz,
                bandpass_high_hz=self.loader_cfg.bandpass_high_hz,
                bandpass_order=self.loader_cfg.bandpass_order,
            )
        return (
            torch.from_numpy(window.astype(np.float32, copy=False)),
            torch.from_numpy(self.site_feats[i]),
            torch.tensor(self.magnitudes[i], dtype=torch.float32),
            torch.tensor(self.aux_log_fc[i], dtype=torch.float32),
            torch.tensor(self.aux_log_tau_c[i], dtype=torch.float32),
            torch.tensor(self.aux_log_pd[i], dtype=torch.float32),
        )

    def magnitudes_array(self) -> np.ndarray:
        return self.magnitudes.copy()


def make_stratified_sampler(
    magnitudes: np.ndarray,
    bin_width: float = 1.0,
    cap: float = 20.0,
    seed: int = 0,
) -> WeightedRandomSampler:
    """WeightedRandomSampler that oversamples rare magnitude bins up to
    `cap`x relative to the most populated bin. Caller is responsible for
    passing in the training-split magnitudes (never val/test)."""
    weights = bin_balanced_weights(magnitudes, bin_width=bin_width, cap=cap)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
