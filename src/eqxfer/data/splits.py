"""California geographic holdout + event-ID-grouped train/val split.

Hard invariants (enforced by tests/test_splits_no_leakage.py):
- No event_id appears in more than one split.
- No station appears in both train and test.
- Same (metadata, config) always yields the same splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SplitConfig


@dataclass(frozen=True)
class SplitResult:
    train: list[str]
    val: list[str]
    test: list[str]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _in_test_box(df: pd.DataFrame, cfg: SplitConfig, prefix: str) -> pd.Series:
    lat = df[f"{prefix}latitude"]
    lon = df[f"{prefix}longitude"]
    return (
        lat.between(cfg.test_lat_min, cfg.test_lat_max)
        & lon.between(cfg.test_lon_min, cfg.test_lon_max)
    )


def _california_ridgecrest(metadata: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    event_in_box = _in_test_box(metadata, cfg, prefix="source_")
    test_df = metadata[event_in_box]
    test_stations = set(test_df["receiver_code"].unique())

    remainder = metadata[
        ~event_in_box & ~metadata["receiver_code"].isin(test_stations)
    ].copy()

    unique_events = remainder["source_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(cfg.seed)
    shuffled = rng.permutation(unique_events)
    n_val = int(round(len(shuffled) * cfg.val_fraction))
    val_events = set(shuffled[:n_val].tolist())
    train_events = set(shuffled[n_val:].tolist())

    train = remainder.loc[
        remainder["source_id"].isin(train_events), "trace_name"
    ].tolist()
    val = remainder.loc[
        remainder["source_id"].isin(val_events), "trace_name"
    ].tolist()
    test = test_df["trace_name"].tolist()
    return SplitResult(train=train, val=val, test=test)


def _event_grouped(metadata: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    """Event-ID-grouped train/val/test split — no geographic holdout box.

    Used for Greece/Chile where total trace counts are too small (21k/5.1k) to
    carve out a geographic sub-region while preserving enough events in
    train. Still enforces the core hard rule (no event crosses splits); also
    prevents train↔test station leakage by reassigning any station that
    ends up in test to test-only."""
    rng = np.random.default_rng(cfg.seed)
    unique_events = metadata["source_id"].drop_duplicates().tolist()
    shuffled = rng.permutation(unique_events)

    n = len(shuffled)
    n_test = int(round(n * cfg.test_fraction))
    n_val = int(round(n * cfg.val_fraction))
    test_events = set(shuffled[:n_test].tolist())
    val_events = set(shuffled[n_test : n_test + n_val].tolist())
    train_events = set(shuffled[n_test + n_val :].tolist())

    test_df = metadata[metadata["source_id"].isin(test_events)]
    test_stations = set(test_df["receiver_code"].unique())
    remainder = metadata[~metadata["receiver_code"].isin(test_stations)].copy()

    train = remainder.loc[
        remainder["source_id"].isin(train_events), "trace_name"
    ].tolist()
    val = remainder.loc[
        remainder["source_id"].isin(val_events), "trace_name"
    ].tolist()
    test = test_df["trace_name"].tolist()
    return SplitResult(train=train, val=val, test=test)


def make_splits(metadata: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    if cfg.strategy == "california_ridgecrest":
        return _california_ridgecrest(metadata, cfg)
    if cfg.strategy == "event_grouped":
        return _event_grouped(metadata, cfg)
    raise ValueError(f"unknown split strategy: {cfg.strategy}")


def few_shot_event_sample(
    metadata: pd.DataFrame,
    n_events: int,
    seed: int,
) -> list[str]:
    """Return trace names from `n_events` distinct events, chosen uniformly at
    random. Used for few-shot target-region fine-tuning."""
    rng = np.random.default_rng(seed)
    unique_events = metadata["source_id"].drop_duplicates().tolist()
    if n_events >= len(unique_events):
        chosen = unique_events
    else:
        idx = rng.choice(len(unique_events), size=n_events, replace=False)
        chosen = [unique_events[i] for i in idx]
    return metadata.loc[metadata["source_id"].isin(set(chosen)), "trace_name"].tolist()
