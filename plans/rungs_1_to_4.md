# Implementation Plan: Rungs 1–4 (California in-region baselines)

**Scope of this plan.** California only. Rungs 1–4 establish the in-region
baselines that every later claim is judged against. Transfer to Greece and
Chile lives at rung 6 and is explicitly out of scope here.

**Status of this plan.** Approved. All nine open questions resolved;
canonical decisions now live in CLAUDE.md under "Data processing
pipeline" and "Definition of done." Summary below for quick reference.

---

## Resolved decisions (canonical versions in CLAUDE.md)

| # | Decision | Value |
|---|---|---|
| Q1 | Magnitude floor | **M ≥ 3.0** |
| Q2 | Magnitude scales | **`ml` OR `mw`** (California) |
| Q3 | SNR threshold | **≥ 10 dB on all 3 components** |
| Q4 | California holdout box | **35.3°–36.0°N, -118.0° to -117.0°W** (Ridgecrest/Coso) |
| Q5 | Per-rung ±0.3 targets | **1: 55–65%, 2: 60–70%, 3: 65–75%, 4: 68–78%** |
| Q6 | 5th physics feature | **τp_max (Nakamura predominant period)** |
| Q7 | Preprocessing | **Detrend + 0.075–25 Hz Butterworth bandpass, zero-phase** |
| Q8 | Class imbalance | **Weighted sampling, 1.0-wide bins, 20× cap, all 4 rungs** |
| Q9 | Feature caching | **Cache to `data/processed/`, hash-keyed by config** |

---

## Shared infrastructure (build in this order)

These are the foundations. Every rung sits on top of them. If any of
these are wrong, every rung is silently wrong.

### S1. STEAD loader — `src/eqxfer/data/stead_loader.py`

- Reads `data/raw/stead/merged.csv` + `merged.hdf5`.
- Applies filters: earthquake_local only, California bounding box,
  magnitude floor (from Q1), magnitude type filter (from Q2), SNR
  filter (from Q3), trace has ≥ p_arrival_sample + 500 samples available.
- Reorders components from STEAD's native ENZ → ZNE (CLAUDE.md spec).
- Returns a `pd.DataFrame` of metadata plus a lazy accessor for the
  (3, 500) waveform window. HDF5 is not loaded into memory.
- Emits a config hash (pydantic config → SHA256) that downstream
  modules use as cache key.

### S2. Splits — `src/eqxfer/data/splits.py`

- Pure function: `make_splits(loader_config, seed) -> (train_ids, val_ids, test_ids)`.
- Implements the geographic holdout (Q4) + event-grouped train/val split.
- Deterministic for a given (loader_config_hash, seed).
- Returns lists of STEAD trace_names, one per split.

### S3. Waveform preprocessing — `src/eqxfer/features/waveform.py`

- Pure functions: `detrend`, `bandpass` (from Q7), `window_from_p`.
- Input: (3, N) raw trace + p_arrival_sample.
- Output: (3, 500) ZNE window.
- No amplitude normalization. Assert output has the same amplitude
  scale as input (regression test).

### S4. Physics features — `src/eqxfer/features/pd_tauc.py`

- Pure functions: `compute_pd(window)`, `compute_tauc(window)`,
  `compute_pgv(window)`, `compute_cav(window)`, `compute_taup_max(window)`.
- Input: (3, 500) ZNE window at 100 Hz.
- Output: scalar per feature, per trace (use Z component for classical
  EEW features; PGV and CAV are magnitude of 3-component vector).
- No silent defaults — if Pd can't be computed (e.g., all zeros), raise.
- Cache results to `data/processed/physics_features_<hash>.parquet`.

### S5. Preprocessing cache — `src/eqxfer/data/filters.py`

- Given loader config + seed, return a `StreamedDataset` yielding
  `(waveform, physics_features, magnitude, metadata)` tuples from
  cached parquet + HDF5.
- First run builds the cache; subsequent runs read it.

### S6. Geographic-split leakage tests — `tests/test_splits_no_leakage.py`

- Test: `set(train_events) ∩ set(test_events) == ∅`.
- Test: `set(train_stations) ∩ set(test_stations) == ∅`.
- Test: train bbox and test bbox don't overlap (if geographic holdout used).
- Test: re-running `make_splits(same_config, same_seed)` gives identical splits.

### S7. Sampling logic — `src/eqxfer/training/samplers.py`

- `MagnitudeBalancedSampler`: PyTorch `WeightedRandomSampler` with
  per-bin inverse-frequency weights capped at 20× relative imbalance.
- For linear models (rungs 1–2), provide weighted-lstsq with the same
  effective weights.

### S8. Metrics — `src/eqxfer/evaluation/metrics.py`

- `compute_metric_panel(y_true, y_pred) -> MetricPanel`.
- Fields: MAE, RMSE, bias, acc_0p3, acc_0p5, acc_1p0 (overall + per
  magnitude bin).
- Bootstrap 95% CIs on everything (1000 resamples; configurable).
- Writes one row to `experiments/results.csv`; full per-run artifacts
  (including bootstrap samples) to `experiments/<exp_id>/metrics.json`.

### S9. Results logger — `src/eqxfer/evaluation/logger.py`

- Writes one row to `experiments/results.csv` per `evaluate.py` invocation.
- Captures: git SHA, config hash, seed, rung, model name, timestamp.
- Appends, doesn't overwrite.

**Build order**: S1 → S2, S3, S6 (in parallel) → S4 → S5 → S7 → S8 →
S9. Nothing depends on S7 until rung 3. S4/S5 not needed until rung 2.

---

## Rung 1: Pd-only log-linear regression

**Files**
- `src/eqxfer/models/pd_linear.py`: `PdLinear` class wrapping weighted lstsq.
- `configs/baseline_pd.yaml`: config including loader filters, seed, holdout box.
- `scripts/train.py` invoked with `--config configs/baseline_pd.yaml --rung 1`.

**Features**: `log10(Pd)` only. Pd = peak displacement on Z component
in the 5s window (integrate velocity → displacement, take max absolute
value).

**Model**: `M_pred = a · log10(Pd) + b`. Two parameters total. Fit via
weighted least-squares using bin-balanced weights (20× cap).

**Training**: single lstsq call. No epochs.

**Evaluation**: apply to test split, compute metric panel via S8.

**Metrics reported**: full panel (MAE, RMSE, bias, ±0.3/0.5/1.0,
magnitude-binned).

**Target ±0.3 accuracy**: 55–65% (per Q5).

---

## Rung 2: Pd + τc linear regression

**Files**
- `src/eqxfer/models/pd_linear.py`: `PdTaucLinear` class (alongside `PdLinear`).
- `configs/baseline_pd_tauc.yaml`.

**Features**: `[log10(Pd), log10(τc)]`. τc computed per Kanamori 2005
on Z component over the 5s window (integrate for displacement and
velocity² numerator / displacement² denominator ratio).

**Model**: `M_pred = a·log10(Pd) + b·log10(τc) + c`.

**Training**: weighted lstsq, same weights as rung 1.

**Evaluation**: identical protocol.

**Target ±0.3 accuracy**: 60–70%.

---

## Rung 3: Raw-waveform 1D CNN

**Files**
- `src/eqxfer/models/raw_cnn.py`: `RawCNN` PyTorch module.
- `src/eqxfer/training/loops.py`: generic training loop (reused by rungs 3, 4, 5).
- `configs/baseline_cnn.yaml`.

**Architecture** (proposed, confirm if you want different):
```
Input: (B, 3, 500) — Z, N, E at 100 Hz
Conv1d(3 → 32, k=7, s=1) + GroupNorm(8) + ReLU
Conv1d(32 → 64, k=7, s=2) + GroupNorm(8) + ReLU        # (B, 64, 247)
Conv1d(64 → 128, k=5, s=2) + GroupNorm(16) + ReLU      # (B, 128, 122)
Conv1d(128 → 128, k=5, s=2) + GroupNorm(16) + ReLU     # (B, 128, 59)
Conv1d(128 → 128, k=3, s=2) + GroupNorm(16) + ReLU     # (B, 128, 29)
AdaptiveAvgPool1d(1) → (B, 128)
Linear(128 → 64) + ReLU + Dropout(0.1)
Linear(64 → 1) → magnitude prediction
```

~180k parameters. No BatchNorm (per CLAUDE.md NOT-do rule). No
attention. GroupNorm preserves amplitude-relevant statistics better
than BatchNorm on waveforms.

**Training**:
- Optimizer: Adam, lr=1e-3
- Schedule: cosine decay to 1e-5 over 100 epochs
- Loss: Huber (δ=0.5) — more robust to label noise than MSE
- Sampler: `MagnitudeBalancedSampler` (S7), 20× cap
- Batch size: 256
- Early stopping: patience 15 on val MAE
- GPU required (CLAUDE.md mandates GPU usage)
- Mixed precision (bfloat16 on Ampere+, fp16 on older)

**Evaluation**: identical protocol.

**Target ±0.3 accuracy**: 65–75%.

---

## Rung 4: CNN + 5 curated physics features

**Files**
- `src/eqxfer/models/physics_cnn.py`: `PhysicsCNN` PyTorch module.
- `configs/baseline_physics_cnn.yaml`.

**Architecture**:
```
Input: waveform (B, 3, 500) + physics (B, 5)
Waveform path: same CNN stack as rung 3 → 128-dim
Physics path: Linear(5 → 16) + ReLU → Linear(16 → 16) + ReLU
Fusion: concat(128, 16) → Linear(144 → 64) + ReLU + Dropout(0.1) → Linear(64 → 1)
```

Physics vector: `[log10(Pd), log10(τc), log10(PGV), log10(CAV), log10(τp_max)]`,
standardized (mean 0, std 1) using training-set statistics saved to the
run directory.

**Training**: identical to rung 3.

**Evaluation**: identical protocol.

**Target ±0.3 accuracy**: 68–78%.

---

## Evaluation protocol (identical across all four rungs)

- **California geographic holdout** (Q4): held out for test. Remaining
  California events split event-ID-grouped 80/10 into train/val.
- **Seeds**: `[0, 1, 2]` for bootstrap-over-seeds variance. Report
  mean ± std across seeds alongside bootstrap CIs within a single seed.
- **Magnitude floor**: per Q1.
- **Magnitude scale**: per Q2.
- **5-second P-wave window**: from STEAD's `p_arrival_sample`,
  ZNE-ordered, no padding, traces shorter than P+5s dropped.
- **Metrics reported per rung per seed**:
  - MAE, RMSE, bias (signed mean error)
  - Accuracy within ±0.3, ±0.5, ±1.0 magnitude units
  - All above, stratified by magnitude bin: M[3–4], M[4–5], M[5–6], M[6+]
    (adjusted per Q1)
  - Bootstrap 95% CIs on every number (1000 resamples)
- **results.csv row per (rung, seed)**: config hash, git SHA, seed,
  rung, model, train/test region, split strategy, n_train, n_test, all
  metrics, notes.
- **Per-run artifacts** to `experiments/<exp_id>/`: config YAML,
  training log JSON, checkpoints (CNN rungs only), predictions CSV,
  metrics JSON with bootstrap samples.

Pass/fail: each rung must land inside its target ±0.3 accuracy range
(Q5) with bootstrap CI overlapping the target midpoint, AND beat the
prior rung by non-overlapping CIs on ±0.3 accuracy. If either fails,
stop and debug.

---

## Dependencies and risk

Each row lists a failure mode, what it would corrupt, and the test
that catches it.

| Failure mode | What it corrupts | Catch |
|---|---|---|
| STEAD loader drops traces silently | All rungs (wrong data) | Unit test: loader applied to known-good subset returns exact expected trace_names; hash assertion |
| Component reorder bug (ENZ → ZNE) | Pd (uses Z), CNN (channel layers) | Unit test: loading a known STEAD trace with known Z-component peak value returns that peak at channel 0 |
| Event ID leaking across splits | Inflated test scores all rungs | S6 test: set intersection empty |
| Station leaking across splits | Inflated test scores all rungs | S6 test: set intersection empty |
| Amplitude normalization sneaking in via library default | Magnitude signal deleted, all rungs collapse to mean | Unit test: load a high-magnitude trace, assert peak amplitude > threshold (will fail if anything normalized) |
| Pd computation in wrong units (e.g., velocity instead of displacement) | Rungs 1, 2, 4; wrong constant in rungs 1–2 (model still fits, but features aren't actually Pd) | Unit test: compute Pd on a synthetic sine wave of known amplitude, assert matches closed-form |
| τc numerically unstable on short windows | Rungs 2, 4 features noisy | Unit test + bootstrap check: τc on 100 synthetic ω² spectrum traces with known corner frequency reproduces corner within 20% |
| Weighted sampler wrong weights | Rungs 3, 4 train on unintended distribution | Runtime assertion: first 10 batches of each epoch have ≥ 1 trace from every populated magnitude bin |
| Mixed-precision loss underflow on small Pd features | Rung 4 training instability | Monitor gradient norms; fall back to fp32 if NaN |
| Feature cache stale | Rungs re-using old cache after config change | Cache key = SHA256 of all loader + filter config fields; miss on any change |
| Results.csv overwritten | Loss of all prior results | Append-only, never open in 'w' mode; git-track the file |

The single biggest silent-corruption risk is the loader + preprocessing
chain (S1 + S3). If we get component ordering wrong, or accidentally
normalize amplitudes, every rung is wrong in a way that looks plausible
(MAE of 0.5-ish). The counter-measure is a hand-picked set of 5 known
STEAD traces with hand-verified expected Pd/PGV/τc values, used as
regression tests.

---

## Sequencing and stop points

1. **Build S1–S9** (shared infrastructure). Stop after S6 leakage tests pass.
2. **Rung 1** end-to-end. Stop. Review `results.csv` row. Pass?
3. **Rung 2**. Stop. Review. Pass?
4. **Rung 3**. Stop. Review. Pass? Inspect training curves for
   overfitting (expected-but-should-be-tolerable given ~20k traces).
5. **Rung 4**. Stop. Review. Pass?
6. **Summary writeup** before rung 5: does rung 4 beat rung 3 by a
   meaningful margin? If not, we have a problem — the physics features
   aren't adding value, which means rung 5's architecture probably
   won't either.
