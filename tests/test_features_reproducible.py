"""Preprocessing + Pd correctness tests. Synthetic inputs with known
answers so we notice if a library change flips a sign, scales an axis,
or reorders channels."""

from __future__ import annotations

import numpy as np
import pytest

from eqxfer.features.pd_tauc import compute_pd, infer_instrument_kind
from eqxfer.features.waveform import (
    bandpass,
    detrend_per_channel,
    extract_window,
    reorder_enz_to_zne,
)


def test_detrend_removes_mean():
    t = np.linspace(0, 5, 500)
    x = np.stack([t * 2.0 + 3.0, t * -1.0 - 4.0, np.sin(t) + 5.0])
    out = detrend_per_channel(x)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-9)


def test_bandpass_preserves_length():
    fs = 100.0
    x = np.random.default_rng(0).normal(size=(3, 500))
    out = bandpass(x, fs, 0.075, 25.0, order=4)
    assert out.shape == x.shape


def test_reorder_enz_to_zne_swaps_e_and_z():
    # STEAD native layout: (N, 3) with channels E, N, Z.
    enz = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.float32)
    zne = reorder_enz_to_zne(enz)
    assert np.array_equal(zne[:, 0], np.array([3, 3, 3]))  # Z first
    assert np.array_equal(zne[:, 1], np.array([2, 2, 2]))  # N middle
    assert np.array_equal(zne[:, 2], np.array([1, 1, 1]))  # E last


def test_reorder_enz_to_zne_rejects_wrong_shape():
    with pytest.raises(ValueError, match="STEAD-native"):
        reorder_enz_to_zne(np.zeros((3, 500)))  # channel-first is wrong convention
    with pytest.raises(ValueError, match="STEAD-native"):
        reorder_enz_to_zne(np.zeros((500, 4)))  # 4 channels: not STEAD
    with pytest.raises(ValueError, match="STEAD-native"):
        reorder_enz_to_zne(np.zeros(500))  # 1D: not a trace


def test_extract_window_raises_on_short_trace():
    x = np.zeros((3, 100))
    with pytest.raises(ValueError, match="too short"):
        extract_window(x, p_sample=50, n_samples=500)


def test_infer_instrument_kind():
    assert infer_instrument_kind("HH") == "velocity"
    assert infer_instrument_kind("BH") == "velocity"
    assert infer_instrument_kind("HN") == "acceleration"
    assert infer_instrument_kind("EN") == "acceleration"
    with pytest.raises(ValueError):
        infer_instrument_kind("XX")
    with pytest.raises(ValueError):
        infer_instrument_kind(None)


def test_pd_on_sine_velocity():
    """Cumulative-integral of A·sin(2πft) with initial=0 is
    (A/(2πf))·(1 − cos(2πft)), whose peak absolute value is A/(πf)."""
    fs = 100.0
    f = 1.0
    amplitude = 5.0
    t = np.arange(500) / fs
    z = amplitude * np.sin(2 * np.pi * f * t)
    window = np.stack([z, np.zeros_like(z), np.zeros_like(z)])
    pd = compute_pd(window, sample_rate_hz=fs, instrument_kind="velocity")
    expected = amplitude / (np.pi * f)
    assert pd == pytest.approx(expected, rel=0.02)


def test_pd_on_cosine_velocity():
    """Integral of A·cos(2πft) with initial=0 is (A/(2πf))·sin(2πft),
    whose peak is A/(2πf). Complements the sine test with a different phase."""
    fs = 100.0
    f = 1.0
    amplitude = 5.0
    t = np.arange(500) / fs
    z = amplitude * np.cos(2 * np.pi * f * t)
    window = np.stack([z, np.zeros_like(z), np.zeros_like(z)])
    pd = compute_pd(window, sample_rate_hz=fs, instrument_kind="velocity")
    expected = amplitude / (2 * np.pi * f)
    assert pd == pytest.approx(expected, rel=0.02)
