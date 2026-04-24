"""Waveform preprocessing: detrend, bandpass, component reorder, window.
Pure functions. No hidden state. Amplitude preserved throughout."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend as sp_detrend, filtfilt


def detrend_per_channel(x: np.ndarray) -> np.ndarray:
    """Remove mean + linear trend per channel. Shape (C, N)."""
    return sp_detrend(x, axis=-1, type="linear")


def bandpass(
    x: np.ndarray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, per channel. Shape (C, N)."""
    nyquist = 0.5 * sample_rate_hz
    wn = (low_hz / nyquist, high_hz / nyquist)
    b, a = butter(order, wn, btype="band")
    return filtfilt(b, a, x, axis=-1)


def reorder_enz_to_zne(trace: np.ndarray) -> np.ndarray:
    """STEAD stores waveforms as (N, 3) in E, N, Z order. Return (N, 3) in Z, N, E.

    Strict about input shape — STEAD's convention is (N_samples, 3_channels),
    and we reject anything else rather than silently guessing.
    """
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError(f"expected STEAD-native (N, 3) shape, got {trace.shape}")
    return trace[:, [2, 1, 0]]


def extract_window(trace_zne_cn: np.ndarray, p_sample: int, n_samples: int) -> np.ndarray:
    """Extract [p_sample, p_sample + n_samples] from a (C, N) ZNE-ordered trace.

    Raises if the trace is too short — hard rule, no zero-padding.
    """
    if trace_zne_cn.ndim != 2 or trace_zne_cn.shape[0] != 3:
        raise ValueError(f"expected (3, N) array, got {trace_zne_cn.shape}")
    end = p_sample + n_samples
    if end > trace_zne_cn.shape[1]:
        raise ValueError(
            f"trace too short: need samples [{p_sample}:{end}], have {trace_zne_cn.shape[1]}"
        )
    return trace_zne_cn[:, p_sample:end]


def preprocess(
    raw_trace_stead: np.ndarray,
    p_sample: int,
    sample_rate_hz: float,
    window_samples: int,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int = 4,
) -> np.ndarray:
    """Full pipeline: STEAD (N, 3) ENZ → preprocessed (3, window_samples) ZNE.

    Order: reorder → transpose to (C, N) → detrend → bandpass → window.
    """
    reordered = reorder_enz_to_zne(raw_trace_stead)  # (N, 3) ZNE
    cn = reordered.T  # (3, N)
    cn = detrend_per_channel(cn)
    cn = bandpass(cn, sample_rate_hz, bandpass_low_hz, bandpass_high_hz, bandpass_order)
    return extract_window(cn, p_sample, window_samples)
