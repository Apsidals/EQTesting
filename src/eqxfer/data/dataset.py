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


class WaveformDataset(Dataset):
    """Returns (waveform, site_feats, magnitude) for one trace.

    waveform:   float32 tensor of shape (3, window_samples), ZNE order
    site_feats: float32 tensor of shape (SITE_FEATURE_DIM,)
    magnitude:  float32 tensor, scalar
    """

    def __init__(
        self,
        loader_cfg: LoaderConfig,
        metadata: pd.DataFrame,
        site_features: pd.DataFrame,
        trace_names: list[str],
    ) -> None:
        self.loader_cfg = loader_cfg
        self.trace_names = list(trace_names)

        idx = metadata.set_index("trace_name").loc[self.trace_names]
        self.p_samples = idx["p_arrival_sample"].astype(int).to_numpy()
        self.magnitudes = idx["source_magnitude"].astype(np.float32).to_numpy()

        feat = site_features.loc[self.trace_names]
        self.site_feats = feat.to_numpy(dtype=np.float32)

        self._hdf5_path = Path(loader_cfg.stead_dir) / loader_cfg.hdf5_name
        self._hdf5: h5py.File | None = None  # opened lazily per worker

    def __len__(self) -> int:
        return len(self.trace_names)

    def _file(self) -> h5py.File:
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self._hdf5_path, "r")
        return self._hdf5

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.trace_names[i]
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
