"""Precomputed waveform cache.

Per-trace preprocessing (detrend + bandpass + window) is deterministic
given a fixed LoaderConfig. Running it every epoch is the main bottleneck
for laptop-class rung-5 training (scipy filtfilt is CPU-heavy), so we
persist the preprocessed (3, window_samples) windows to a memmapped
.npy and epoch N+1 becomes one numpy slice per sample.

Cache layout under the cache_dir:
    waveforms_<key>.npy         # (N, 3, window_samples) float32 memmap
    waveforms_<key>.index.json  # list[str] — trace_name at each row

The key hashes (preprocessing params, sorted trace-name set). Two configs
with the same preprocessing but different filter sets get different files,
so you can cache California and Greece side by side.

Default cache_dir is ~/.cache/eqxfer/waveforms (WSL native ext4). Override
with EQXFER_CACHE_DIR — important when the repo lives on /mnt/c/ since
WSL2's 9P bridge is 10-50x slower than native for random-access reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from ..config import LoaderConfig
from ..features.waveform import preprocess


def default_cache_dir() -> Path:
    override = os.environ.get("EQXFER_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "eqxfer" / "waveforms"


def _cache_key(loader_cfg: LoaderConfig, trace_names: list[str]) -> str:
    h = hashlib.sha256()
    key_dict = {
        "window_samples": loader_cfg.window_samples,
        "sample_rate_hz": loader_cfg.sample_rate_hz,
        "bandpass_low_hz": loader_cfg.bandpass_low_hz,
        "bandpass_high_hz": loader_cfg.bandpass_high_hz,
        "bandpass_order": loader_cfg.bandpass_order,
        "region": loader_cfg.region,
    }
    h.update(json.dumps(key_dict, sort_keys=True).encode())
    # Sorted trace set goes into the key so different filter results get
    # different files instead of pingponging one cache between configs.
    for name in sorted(trace_names):
        h.update(name.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


class WaveformCache:
    def __init__(
        self,
        loader_cfg: LoaderConfig,
        trace_names: list[str],
        cache_dir: Path | None = None,
    ) -> None:
        self.loader_cfg = loader_cfg
        self.trace_names = list(trace_names)
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.key = _cache_key(loader_cfg, self.trace_names)
        self.data_path = self.cache_dir / f"waveforms_{self.key}.npy"
        self.index_path = self.cache_dir / f"waveforms_{self.key}.index.json"
        # Worker-local state — populated by open() after fork/spawn.
        self._mmap: np.ndarray | None = None
        self._trace_to_row: dict[str, int] = {}

    def exists(self) -> bool:
        if not (self.data_path.exists() and self.index_path.exists()):
            return False
        try:
            idx = json.loads(self.index_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return isinstance(idx, list) and set(idx) >= set(self.trace_names)

    def build(self, p_samples: np.ndarray) -> None:
        """(Re)build the cache. `p_samples` is a length-N int array aligned
        with self.trace_names — caller provides these from the metadata
        table so we don't re-query HDF5 for p-arrival picks."""
        if len(p_samples) != len(self.trace_names):
            raise ValueError(
                f"p_samples length {len(p_samples)} != trace_names length "
                f"{len(self.trace_names)}"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.loader_cfg
        hdf5_path = Path(cfg.stead_dir) / cfg.hdf5_name
        if not hdf5_path.exists():
            raise FileNotFoundError(f"STEAD HDF5 not found: {hdf5_path}")

        n = len(self.trace_names)
        tmp_path = self.data_path.with_suffix(".npy.tmp")
        arr = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, 3, cfg.window_samples),
        )
        print(
            f"[cache] building {self.data_path.name} "
            f"({n:,} traces × 3 × {cfg.window_samples} float32 "
            f"= {n * 3 * cfg.window_samples * 4 / 1024**3:.2f} GB)",
            flush=True,
        )
        t0 = time.time()
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            for i, (name, p) in enumerate(zip(self.trace_names, p_samples)):
                raw = np.asarray(data[name][:], dtype=np.float32)
                win = preprocess(
                    raw_trace_stead=raw,
                    p_sample=int(p),
                    sample_rate_hz=cfg.sample_rate_hz,
                    window_samples=cfg.window_samples,
                    bandpass_low_hz=cfg.bandpass_low_hz,
                    bandpass_high_hz=cfg.bandpass_high_hz,
                    bandpass_order=cfg.bandpass_order,
                )
                arr[i] = win.astype(np.float32, copy=False)
                if (i + 1) % 10_000 == 0 or i + 1 == n:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1e-6)
                    eta = (n - (i + 1)) / max(rate, 1e-6)
                    print(
                        f"[cache] {i+1:>7,}/{n:,} "
                        f"({rate:.0f} tr/s, eta {eta/60:.1f} min)",
                        flush=True,
                    )
        arr.flush()
        del arr  # close the memmap before rename
        os.replace(tmp_path, self.data_path)
        self.index_path.write_text(json.dumps(list(self.trace_names)))
        print(
            f"[cache] built {self.data_path} in {(time.time()-t0)/60:.1f} min",
            flush=True,
        )

    def open(self) -> None:
        """Populate worker-local mmap + index. Safe to call repeatedly;
        no-op after first successful call in the process."""
        if self._mmap is not None:
            return
        self._mmap = np.load(self.data_path, mmap_mode="r")
        idx = json.loads(self.index_path.read_text())
        self._trace_to_row = {name: i for i, name in enumerate(idx)}

    def get(self, trace_name: str) -> np.ndarray:
        # Copy off the memmap so downstream torch.from_numpy owns the buffer;
        # returning the mmap view directly can hold file pages live longer
        # than the DataLoader batch lifetime.
        assert self._mmap is not None, "WaveformCache.open() not called"
        row = self._trace_to_row[trace_name]
        return np.array(self._mmap[row], dtype=np.float32)
