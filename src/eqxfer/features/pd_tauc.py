"""Classical EEW features. Pd, τc, PGV, plus a simple spectral corner-
frequency estimator used as a probe target for the split architecture's
universal encoder.

Per CLAUDE.md: no silent default fallbacks. Raise on malformed inputs."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid

InstrumentKind = Literal["velocity", "acceleration"]


def infer_instrument_kind(receiver_type: object) -> InstrumentKind:
    """STEAD's receiver_type is a 2-char channel-code prefix (e.g. 'HH', 'HN').
    Second char encodes instrument: H/B/L/S = seismometer (velocity),
    N/G = accelerometer (acceleration). Anything else → raise."""
    if not isinstance(receiver_type, str) or len(receiver_type) < 2:
        raise ValueError(f"unusable receiver_type: {receiver_type!r}")
    kind = receiver_type[1].upper()
    if kind in ("H", "B", "L", "S"):
        return "velocity"
    if kind in ("N", "G"):
        return "acceleration"
    raise ValueError(f"unknown instrument kind in receiver_type={receiver_type!r}")


def compute_pd(
    window_zne: np.ndarray,
    sample_rate_hz: float,
    instrument_kind: InstrumentKind,
) -> float:
    """Peak absolute displacement on the Z channel over the 5s window.

    window_zne shape: (3, N), channel 0 = Z (vertical).
    Wu & Kanamori 2005 style. For velocity inputs, single integration.
    For acceleration inputs, double integration.
    """
    if window_zne.ndim != 2 or window_zne.shape[0] != 3:
        raise ValueError(f"expected (3, N) ZNE window, got {window_zne.shape}")
    z = window_zne[0]
    dt = 1.0 / sample_rate_hz
    if instrument_kind == "velocity":
        displacement = cumulative_trapezoid(z, dx=dt, initial=0.0)
    elif instrument_kind == "acceleration":
        velocity = cumulative_trapezoid(z, dx=dt, initial=0.0)
        displacement = cumulative_trapezoid(velocity, dx=dt, initial=0.0)
    else:
        raise ValueError(f"unknown instrument_kind={instrument_kind}")
    peak = float(np.max(np.abs(displacement)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"Pd computation produced non-positive or non-finite value: {peak}")
    return peak


def _to_velocity_displacement(
    z: np.ndarray,
    sample_rate_hz: float,
    instrument_kind: InstrumentKind,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (velocity, displacement) on Z channel. Integrates as needed."""
    dt = 1.0 / sample_rate_hz
    if instrument_kind == "velocity":
        velocity = z
        displacement = cumulative_trapezoid(z, dx=dt, initial=0.0)
    elif instrument_kind == "acceleration":
        velocity = cumulative_trapezoid(z, dx=dt, initial=0.0)
        displacement = cumulative_trapezoid(velocity, dx=dt, initial=0.0)
    else:
        raise ValueError(f"unknown instrument_kind={instrument_kind}")
    return velocity, displacement


def compute_tau_c(
    window_zne: np.ndarray,
    sample_rate_hz: float,
    instrument_kind: InstrumentKind,
) -> float:
    """Nakamura 1988 τc on the Z channel — a characteristic period that
    scales with rupture size. τc = 2π sqrt(integral(d²) / integral(v²)).

    Used as a probe target for rupture duration."""
    if window_zne.ndim != 2 or window_zne.shape[0] != 3:
        raise ValueError(f"expected (3, N) ZNE window, got {window_zne.shape}")
    z = window_zne[0]
    velocity, displacement = _to_velocity_displacement(z, sample_rate_hz, instrument_kind)
    dt = 1.0 / sample_rate_hz
    num = float(np.trapezoid(displacement**2, dx=dt))
    den = float(np.trapezoid(velocity**2, dx=dt))
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0 or num <= 0.0:
        raise ValueError(f"τc computation: num={num}, den={den}")
    return float(2.0 * np.pi * np.sqrt(num / den))


def compute_pgv(
    window_zne: np.ndarray,
    sample_rate_hz: float,
    instrument_kind: InstrumentKind,
) -> float:
    """Peak absolute ground velocity on the Z channel."""
    if window_zne.ndim != 2 or window_zne.shape[0] != 3:
        raise ValueError(f"expected (3, N) ZNE window, got {window_zne.shape}")
    z = window_zne[0]
    velocity, _ = _to_velocity_displacement(z, sample_rate_hz, instrument_kind)
    peak = float(np.max(np.abs(velocity)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"PGV computation non-positive/non-finite: {peak}")
    return peak


def compute_corner_frequency(
    window_zne: np.ndarray,
    sample_rate_hz: float,
    instrument_kind: InstrumentKind,
    low_hz: float = 0.1,
    high_hz: float = 25.0,
) -> float:
    """Crude Brune-model-style corner frequency from the Z-component
    displacement spectrum. The corner is estimated as the frequency where
    the amplitude spectrum drops to 1/sqrt(2) of its low-frequency plateau.

    This is a deliberately simple estimator — it's a probe target, not a
    finished physics product. What we care about is whether the universal
    encoder's embedding can regress onto it."""
    if window_zne.ndim != 2 or window_zne.shape[0] != 3:
        raise ValueError(f"expected (3, N) ZNE window, got {window_zne.shape}")
    z = window_zne[0]
    _, displacement = _to_velocity_displacement(z, sample_rate_hz, instrument_kind)
    n = displacement.shape[-1]
    spectrum = np.abs(np.fft.rfft(displacement))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        raise ValueError("no frequency bins in [low_hz, high_hz]")
    f = freqs[mask]
    s = spectrum[mask]
    # Low-frequency plateau estimate = median of the bottom third of the band.
    n_low = max(1, len(f) // 3)
    plateau = float(np.median(s[:n_low]))
    if plateau <= 0.0 or not np.isfinite(plateau):
        raise ValueError(f"spectrum plateau non-positive/non-finite: {plateau}")
    threshold = plateau / np.sqrt(2.0)
    below = np.where(s < threshold)[0]
    if len(below) == 0:
        return float(f[-1])
    return float(f[below[0]])
