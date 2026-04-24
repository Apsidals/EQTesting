"""Batch samplers.

EventGroupedBatchSampler produces batches where co-source traces sit
together, so the training loop can compute within-event variance of the
physics embedding (L_sep, see SeparationLossConfig). Each batch is a
concatenation of `events_per_batch` blocks of size `stations_per_event`;
reshape at train time to (events, stations, embed_dim) and variance is
over dim=1.

Event-level magnitude balancing replaces the per-trace WeightedRandomSampler
used before. The sampler_cap parameter now caps how much a rare-magnitude
*event* is oversampled relative to the most-populated magnitude bin.
Running two imbalance mechanisms at once (sampler + per-sample loss
weight) was what CLAUDE.md's hard rule #8 warns against — this subsumes
both into one place.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

from ..models.pd_linear import bin_balanced_weights


class EventGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        event_ids: np.ndarray,
        magnitudes: np.ndarray,
        events_per_batch: int,
        stations_per_event: int,
        min_stations_per_event: int = 2,
        bin_width: float = 1.0,
        sampler_cap: float = 3.0,
        n_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        if len(event_ids) != len(magnitudes):
            raise ValueError(
                f"event_ids length {len(event_ids)} != magnitudes length "
                f"{len(magnitudes)}"
            )
        self.events_per_batch = events_per_batch
        self.stations_per_event = stations_per_event
        self.batch_size = events_per_batch * stations_per_event
        self.rng = np.random.default_rng(seed)

        event_to_indices: dict[object, list[int]] = defaultdict(list)
        for i, eid in enumerate(event_ids):
            event_to_indices[eid].append(i)

        eligible = [
            eid
            for eid, idxs in event_to_indices.items()
            if len(idxs) >= min_stations_per_event
        ]
        if not eligible:
            raise ValueError(
                f"no events have >= {min_stations_per_event} stations — "
                f"EventGroupedBatchSampler would emit empty batches"
            )
        self.eligible_events = eligible
        self.event_to_indices: dict[object, np.ndarray] = {
            eid: np.asarray(event_to_indices[eid], dtype=np.int64)
            for eid in eligible
        }

        # Per-event mean magnitude → bin-balanced event sampling weights.
        event_mean_mag = np.array(
            [float(magnitudes[self.event_to_indices[eid]].mean()) for eid in eligible],
            dtype=np.float64,
        )
        raw_w = bin_balanced_weights(
            event_mean_mag, bin_width=bin_width, cap=sampler_cap
        )
        self.event_weights = raw_w / raw_w.sum()

        # Account for dataset shrinkage: only traces in eligible events are
        # reachable. Batches/epoch defaults to "cover the eligible set once".
        n_eligible_traces = sum(len(v) for v in self.event_to_indices.values())
        self._n_eligible_traces = n_eligible_traces
        self.n_batches = (
            int(n_batches)
            if n_batches is not None
            else max(1, n_eligible_traces // self.batch_size)
        )

    def n_eligible_traces(self) -> int:
        return self._n_eligible_traces

    def __iter__(self):
        events = np.asarray(self.eligible_events, dtype=object)
        for _ in range(self.n_batches):
            chosen = self.rng.choice(
                len(events),
                size=self.events_per_batch,
                replace=True,
                p=self.event_weights,
            )
            batch: list[int] = []
            for ei in chosen:
                idxs = self.event_to_indices[events[ei]]
                k = len(idxs)
                # With replacement when the event has fewer stations than we
                # want. Duplicates contribute 0 to within-event variance, so
                # events with more stations produce a stronger L_sep signal.
                replace = k < self.stations_per_event
                sampled = self.rng.choice(
                    idxs, size=self.stations_per_event, replace=replace
                )
                batch.extend(int(x) for x in sampled)
            yield batch

    def __len__(self) -> int:
        return self.n_batches
