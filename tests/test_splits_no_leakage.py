"""Leakage tests for California Ridgecrest split. Uses synthetic metadata
so we don't depend on STEAD being present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqxfer.config import SplitConfig
from eqxfer.data.splits import make_splits


def _synthetic_metadata(n_events: int = 200, stations_per_event: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for event_idx in range(n_events):
        in_test_box = event_idx % 5 == 0
        if in_test_box:
            lat = rng.uniform(35.3, 36.0)
            lon = rng.uniform(-118.0, -117.0)
            station_prefix = "TEST"
        else:
            lat = rng.uniform(33.0, 40.0)
            lon = rng.uniform(-123.0, -118.5)
            station_prefix = "TRN"
        for station_idx in range(stations_per_event):
            rows.append(
                {
                    "trace_name": f"E{event_idx:04d}_S{station_idx:02d}",
                    "source_id": f"evt{event_idx:04d}",
                    "source_latitude": lat,
                    "source_longitude": lon,
                    "receiver_code": f"{station_prefix}{station_idx:02d}_{event_idx % 11:02d}",
                    "receiver_latitude": lat + rng.normal(scale=0.05),
                    "receiver_longitude": lon + rng.normal(scale=0.05),
                    "source_magnitude": float(rng.uniform(3.0, 6.0)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def metadata() -> pd.DataFrame:
    return _synthetic_metadata()


@pytest.fixture
def split(metadata: pd.DataFrame):
    return make_splits(metadata, SplitConfig(seed=0))


def _lookup(metadata: pd.DataFrame, trace_names: list[str], col: str) -> set:
    return set(metadata.loc[metadata["trace_name"].isin(trace_names), col])


def test_no_event_leakage(metadata, split):
    train_events = _lookup(metadata, split.train, "source_id")
    val_events = _lookup(metadata, split.val, "source_id")
    test_events = _lookup(metadata, split.test, "source_id")
    assert train_events.isdisjoint(test_events), "event leaked from train into test"
    assert val_events.isdisjoint(test_events), "event leaked from val into test"
    assert train_events.isdisjoint(val_events), "event leaked between train and val"


def test_no_station_train_test_leakage(metadata, split):
    train_stations = _lookup(metadata, split.train, "receiver_code")
    test_stations = _lookup(metadata, split.test, "receiver_code")
    assert train_stations.isdisjoint(test_stations), "station leaked from train into test"


def test_deterministic(metadata):
    a = make_splits(metadata, SplitConfig(seed=0))
    b = make_splits(metadata, SplitConfig(seed=0))
    assert a.train == b.train
    assert a.val == b.val
    assert a.test == b.test


def test_different_seed_changes_trainval_split(metadata):
    a = make_splits(metadata, SplitConfig(seed=0))
    b = make_splits(metadata, SplitConfig(seed=1))
    assert a.test == b.test
    assert set(a.train) != set(b.train) or set(a.val) != set(b.val)


def test_every_trace_assigned_or_dropped(metadata, split):
    assigned = set(split.train) | set(split.val) | set(split.test)
    all_names = set(metadata["trace_name"])
    assert assigned.issubset(all_names)
