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
    """Event-ID-grouped train/val/test split for data-scarce target regions.

    Bipartite partition: stations are split into disjoint train/val/test
    pools (weighted by trace count so each pool holds roughly the target
    fraction of traces), events are assigned to the pool where the
    majority of their traces' stations live (ties → train → val → test,
    in that order to preserve data in the larger splits), and a trace is
    kept iff its event's split and its station's split agree.

    This preserves both hard rules — no event appears in two splits, no
    station appears in train AND test — while losing an order of magnitude
    fewer traces than the old "drop all test-station traces from train"
    approach. That old approach is catastrophic for small, densely
    connected networks (Chile has ~200 events over ~80 stations, and test
    events touch most of the network — the old algorithm collapsed a 1,634-
    trace filtered set down to 50 train traces).

    California does not use this strategy; it uses california_ridgecrest
    with a geographic test box and doesn't suffer the same data loss."""

    rng = np.random.default_rng(cfg.seed)

    # Station-level partition. Weight by trace count so each split gets
    # ~target fraction of traces rather than ~target fraction of stations
    # (stations vary wildly in how many traces they record).
    station_counts = metadata["receiver_code"].value_counts()
    shuffled = rng.permutation(station_counts.index.to_numpy())
    total_traces = int(station_counts.sum())
    n_test_target = int(round(total_traces * cfg.test_fraction))
    n_val_target = int(round(total_traces * cfg.val_fraction))

    station_split: dict[str, str] = {}
    acc_test = acc_val = 0
    for s in shuffled:
        c = int(station_counts[s])
        # Fill test first, then val, then everything else to train.
        # Greedy, so the last station assigned to test may overshoot
        # n_test_target by its trace count — acceptable slop.
        if acc_test < n_test_target:
            station_split[str(s)] = "test"
            acc_test += c
        elif acc_val < n_val_target:
            station_split[str(s)] = "val"
            acc_val += c
        else:
            station_split[str(s)] = "train"

    # Event-level assignment: majority-station-split per event.
    df = metadata.assign(
        _sta_split=metadata["receiver_code"].astype(str).map(station_split),
    )
    per_event = (
        df.groupby("source_id")["_sta_split"].value_counts().unstack(fill_value=0)
    )
    for col in ("train", "val", "test"):
        if col not in per_event.columns:
            per_event[col] = 0
    # Column order determines idxmax tie-breaking: train > val > test.
    # Preserves the most data in the largest split when an event's traces
    # are evenly distributed across pools.
    per_event = per_event[["train", "val", "test"]]
    event_split_series = per_event.idxmax(axis=1)
    event_split = event_split_series.to_dict()

    df = df.assign(_evt_split=df["source_id"].map(event_split))

    kept_mask = df["_sta_split"] == df["_evt_split"]
    kept = df.loc[kept_mask]
    n_dropped = int((~kept_mask).sum())

    train = kept.loc[kept["_sta_split"] == "train", "trace_name"].tolist()
    val = kept.loc[kept["_sta_split"] == "val", "trace_name"].tolist()
    test = kept.loc[kept["_sta_split"] == "test", "trace_name"].tolist()

    n_stations = len(station_split)
    n_train_st = sum(1 for v in station_split.values() if v == "train")
    n_val_st = sum(1 for v in station_split.values() if v == "val")
    n_test_st = sum(1 for v in station_split.values() if v == "test")
    print(
        f"[splits] event_grouped (bipartite): "
        f"{n_stations} stations → {n_train_st}/{n_val_st}/{n_test_st} "
        f"train/val/test. "
        f"Traces: {len(train):,}/{len(val):,}/{len(test):,}, "
        f"{n_dropped:,} dropped to preserve non-overlap "
        f"({100*n_dropped/max(1,len(df)):.1f}%).",
        flush=True,
    )

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
