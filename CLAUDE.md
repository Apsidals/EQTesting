# Earthquake Cross-Tectonic Magnitude Transfer

## Scientific hypothesis

**Earthquake source physics is universal; site response is regional.**

If we build an architecture that explicitly separates these two concerns,
a model pretrained in one tectonic regime should transfer to a completely
different regime with minimal adaptation data — because the universal
physics branch has learned something real about how faults rupture, not
just how California faults rupture.

Concretely:
- A **Universal Physics Encoder** consumes raw 3-component P-wave
  waveforms and nothing else. Its job is to learn source-side invariants:
  corner frequency, stress drop proxies, rupture growth rate, radiated
  energy characteristics. These quantities are physics, not geography.
- A **Regional Site Context Encoder** consumes station-local geological
  properties (Vs30, crustal Vp, site class, basin indicators). Its job
  is to learn how the local site modifies what we observe at that
  receiver.
- A **Fusion Layer** combines them to predict magnitude.

**The hypothesis is falsifiable.** If transfer fails badly to Greece/Chile
despite the architectural separation, the hypothesis is wrong — either
the physics branch isn't learning physics, or the separation is leaky.
Both are publishable results. Negative results count.

## The transfer protocol

- **Pretrain**: California only. Strike-slip dominant, San Andreas system,
  dense seismic network. Training magnitude range M ≥ 3.0 (see Data
  processing pipeline). STEAD's California subset contains no M6.5+
  events — the model is trained on small-to-moderate events and
  extrapolated to larger ones at inference. Document this limitation.
- **Zero-shot test**: Greece (Hellenic subduction + normal faulting)
  and Chile (megathrust subduction). Japan is excluded (zero coverage
  in STEAD) and Turkey is excluded (~1.2k traces, too thin).
- **Few-shot test**: fine-tune the Regional Site Context Encoder only,
  with 100 / 500 / 2000 target-region events. Universal encoder stays
  frozen. Measure how quickly each region adapts.
- **Full retrain baseline**: for each target region, also train a model
  from scratch on that region's data alone. If our transfer model can't
  beat the from-scratch baseline at low data budgets, the transfer story
  isn't working.

## What "universal physics" means operationally

This is where v1 went wrong. "Universal" cannot mean "whatever the CNN
learns from California waveforms" — that's just California features with
a universal label slapped on. To earn the label, the physics branch must:

1. **See no regional identifiers.** No lat/lon, no station ID, no
   regional label, no Vs30, no distance. Raw 3-component waveform only.
2. **Have its outputs probed for physics-interpretable quantities.**
   We should be able to regress corner frequency, stress drop, and
   rupture duration from its embeddings with reasonable R². If we can't,
   it's not learning physics, it's learning shortcuts.
3. **Transfer its embedding distribution.** Source and target embeddings
   should overlap in t-SNE/UMAP projections. If California embeddings
   cluster separately from Chile embeddings, the encoder has learned
   California, not physics.

Include these diagnostics as first-class evaluation, not afterthoughts.

## Input specification

Fixed for the entire project. If this changes, it is a paper-wide
change and every result must be regenerated.

- **Window length**: 5.0 seconds, fixed. No multi-window training, no
  variable-length inference. The model consumes a 5s window and emits
  one magnitude estimate.
- **Window start**: STEAD's labeled P-arrival sample (`p_arrival_sample`).
  Use STEAD's pick as ground truth; do not re-pick. For target regions
  without STEAD, use whatever pick the host catalog provides and
  document it.
- **Sample rate**: 100 Hz (STEAD standard). 5s × 100 Hz = 500 samples
  per component. No resampling unless a target region is non-100 Hz,
  in which case resample to 100 Hz at load time.
- **Components**: 3-component in **ZNE order** (vertical, north, east) —
  SeisBench's default, more standard in seismology. Channel 0 = Z,
  channel 1 = N, channel 2 = E. Reorder at load for any region that
  doesn't deliver in ZNE; never reorder at model input. All downstream
  code (feature extractors, per-channel model layers, diagnostic
  plots) assumes this ordering.
- **Shape into the model**: `(batch, 3, 500)` with `dimension_order="NCW"`
  — matches SeisBench's default. Batch = N, Channel (Z/N/E) = C,
  Waveform samples = W.
- **No pre-P padding.** Window is strictly [P, P+5s]. Pre-P noise is
  not used.
- **If a trace is shorter than P+5s, drop it.** Do not zero-pad. Short
  traces are silently biased toward small events (close stations of
  big events clip early) and padding teaches the model that zeros =
  small magnitude.
- **No amplitude normalization.** Per hard rule in the NOT-do list —
  magnitude is amplitude.

## Data processing pipeline

Fixed project-wide. Every rung, every region, uses this pipeline.
Changing any of these requires re-running the entire ladder.

### Filters (applied at load time, in order)

1. **Trace category**: `earthquake_local` only. Noise traces excluded
   from training and evaluation.
2. **Magnitude floor**: **M ≥ 3.0**. Below this, STEAD is dominated by
   microquakes that are operationally irrelevant to EEW and bias the
   trained model toward the mean. Chosen over M ≥ 4.0 because the
   California training set at M ≥ 4.0 is only ~4k traces — insufficient
   for deep models — and over M ≥ 2.5 to avoid pulling in sub-threshold
   events whose magnitudes are noisy.
3. **Magnitude scales**: **`ml` OR `mw`** for California. `ml` (~70%
   of STEAD labels) dominates small events; `mw` is the physically
   correct label at M ≥ 5 where `ml` saturates. Including `mw`
   recovers the 2019 Ridgecrest M6.4/M7.1 sequence and other large
   California events that the catalog reports only in `mw`. Mixed
   scales below M5 introduce minor label noise — acceptable given the
   paper's focus on source-physics invariants rather than scale
   fidelity. Target regions use whatever `ml`/`mw` mix their host
   catalog provides; document the scale split per region.
4. **SNR**: ≥ **10 dB on all 3 components**. Lenient enough to keep
   most traces, strict enough to exclude clearly bad recordings.
   Uses STEAD's `snr_db` field.
5. **Window availability**: trace must have ≥ `p_arrival_sample + 500`
   samples. Shorter traces dropped (no zero-padding — hard rule).

### Waveform preprocessing (applied after filters, before windowing)

1. **Detrend**: remove mean + linear trend per component. Required
   because STEAD traces have small DC offsets that accumulate during
   displacement integration.
2. **Bandpass filter**: 4-pole Butterworth, 0.075–25 Hz (STEAD's
   documented passband). Applied per component, zero-phase
   (`filtfilt`).
3. **Component reorder**: ENZ (STEAD native) → ZNE (project standard).
4. **Window extraction**: [`p_arrival_sample`, `p_arrival_sample + 500`].
5. **No normalization**. Amplitude preserved through the whole chain.

### Class imbalance handling

- Training uses **magnitude-stratified weighted sampling**: 1.0-wide
  magnitude bins, per-sample weight ∝ 1 / bin_count, capped so the
  rarest bin is oversampled at most **20×** relative to the most
  populated bin. Ties to hard rule #8 ("no loss weight above 20×").
- Linear rungs (1, 2) use weighted least-squares with the same
  effective weights. Deep rungs (3, 4, 5) use PyTorch
  `WeightedRandomSampler`.
- **Evaluation uses the natural distribution**, never weighted.
  Stratified metrics (mae_m3, mae_m4, mae_m5, mae_m6plus) are the
  honest accounting.

### California geographic holdout

- **Test box**: Ridgecrest/Coso region, `35.3°–36.0°N, -118.0° to -117.0°W`.
  All California events inside this box → test set. This region
  contains the 2019 Ridgecrest sequence (M6.4 + M7.1 foreshock/mainshock)
  and substantial aftershock activity, giving us genuine large-event
  coverage in test.
- **Remaining California events** → event-ID-grouped 80/10 train/val
  split. Events don't cross train/val; stations don't cross train/test.
- Enforced by pytest tests in `tests/test_splits_no_leakage.py`.

## Hard rules (non-negotiable)

1. **No label leakage.** No feature derived from magnitude, including
   estimated_magnitude_from_amplitude, reaches the model. Ever.
2. **No hypocentral distance leakage.** Distance from event to station
   is unknown in real-time EEW. It does not enter the model. This
   includes geometric spreading corrections that use true distance.
3. **Geographic test splits, never random.** Events from the same
   earthquake (multiple stations) must not cross train/val. Use a
   geographic holdout box or event-ID-based grouping.
4. **Station leakage check.** A station should not appear in both train
   and test for a given region. Write a pytest test for this.
5. **The physics branch sees no regional information.** If you're
   tempted to feed Vs30 or coordinates to the physics branch "to help
   it," you've just collapsed the separation the whole paper depends on.
6. **Baselines run before any fancy architecture.** See ladder below.
7. **Physics feature count stays under 20** until we have ablation
   evidence that additional features help. v1 had 85 and couldn't tell
   which mattered.
8. **Every run writes a row to experiments/results.csv** with config
   hash, seed, split strategy, target metrics, and git commit SHA.

## Baseline ladder (run in order, never skip)

Each rung must be evaluated on the same splits with the same seeds
before moving up. Log everything.

1. **Pd-only log-linear regression.** One feature (peak displacement),
   one intercept, one slope. This is the operational EEW benchmark
   (Wu & Kanamori 2005 style). If our deep model isn't beating this
   by a clear margin, we have nothing.
2. **Pd + τc linear.** Two features. This is roughly state-of-the-art
   for classical EEW. Second benchmark to beat.
3. **Raw-waveform 1D CNN, no physics features, no site context.** How
   much can the network do with just waveforms?
4. **CNN + 3–5 curated physics features.** Pd, τc, peak velocity,
   cumulative absolute velocity, and maybe one more. This is rung 3
   plus explicit physics guidance.
5. **Split architecture (the hypothesis)**: Universal Physics Encoder
   + Regional Site Encoder + Fusion. This is where our method lives.
6. **Transfer experiments** only start after rung 5 works in-region.

Do not build rung 5 before rung 4 is beating rung 3 convincingly.

## What to explicitly NOT do

- No attention mechanism until it's been ablated against non-attention
  and wins. v1 had attention everywhere with no evidence it helped.
- No BatchNorm on the waveform path. It destroys amplitude scale.
  Use GroupNorm or LayerNorm.
- No amplitude normalization on input waveforms. Magnitude is amplitude.
  If you normalize it away, you've deleted the label signal.
- No per-sample default-feature fallbacks in the feature extractor.
  If extraction fails, fail loudly. Silent defaults hid bugs in v1.
- No "quality filter" heuristics added without an ablation showing
  they improve transfer, not just in-region accuracy.
- No magnitude loss weight above 20x. v1 had max_weight=99999 which
  made rare-bin outliers dominate gradients.

## Evaluation criteria

Every rung reports the same metric panel on the same splits/seeds so
results are comparable across the ladder. All numbers go into
`experiments/results.csv` alongside config hash, seed, and git SHA.

### 1. Core prediction quality (in-region)

- **MAE** — mean absolute error in magnitude units. Headline number.
- **RMSE** — fat-tail indicator. If RMSE ≫ MAE, residuals are skewed.
- **Bias** — mean signed error. Catches systematic over/under-prediction
  (e.g., a model that always predicts ~M4 can post decent MAE and still
  be useless).
- **Accuracy within ±0.3, ±0.5, ±1.0** magnitude units. ±0.3 is the
  operational EEW threshold. ±1.0 is a sanity check that should be
  near 100%.

### 2. Magnitude-stratified metrics (non-negotiable)

Report every metric above, binned by true magnitude:
**M2.5–3.5, 3.5–4.5, 4.5–5.5, 5.5–6.5, 6.5+.**

Two reasons this is mandatory:
1. Small events dominate the dataset. Aggregate MAE is essentially
   "MAE on M3 events" with a correction term.
2. **Large-event underestimation is the failure mode that kills EEW.**
   An M7 registering as M5.2 at t=3s means the wrong area got alerted.
   Report *signed residuals* in the M6+ bin explicitly — reviewers
   will look at this slide first.

### 3. Transfer-specific metrics (rung 6)

- **Transfer gap**: `(MAE_target − MAE_source) / MAE_source`. Small for
  Turkey (sanity), moderate for Greece/Japan, nonzero-but-beats-
  from-scratch for Chile.
- **Few-shot efficiency curve**: MAE vs. n_adaptation_events at
  {0, 100, 500, 2000}. Plot transfer model against from-scratch
  baseline on the same axes.
- **Data efficiency crossover**: how many target-region events does
  from-scratch need to match zero-shot transfer? If ~500, the physics
  encoder is carrying real knowledge.

### 4. Physics-probe metrics (what makes this a physics paper)

These are first-class evaluation, not afterthoughts. They test whether
the Universal Physics Encoder actually learned universal physics.

- **R² on corner frequency** regressed from frozen universal-encoder
  embeddings. Target > 0.5.
- **R² on stress drop** (or a stress-drop proxy) from embeddings.
  Target > 0.5.
- **R² on rupture duration** from embeddings. Target > 0.5.
- **MMD (maximum mean discrepancy)** between California and
  target-region embeddings from the universal branch. Lower = better
  overlap = encoder has not memorized "California."
- **Silhouette score** of region labels on embeddings. Low is good
  (regions mixed in embedding space).
- **UMAP/t-SNE projection** — qualitative figure, but include it.

If R² on corner frequency is < 0.3, the hypothesis is in trouble:
either the encoder didn't learn physics or the embeddings aren't
exposing it. Both are publishable findings, but they change the story.

### 5. Operational EEW metrics (strong additions)

Note: with the fixed 5s window (see "Input specification"), there is
no time-to-estimate curve. If we later add variable-window training,
this is where it would live.

- **Alert-category confusion matrix**: bucket predictions into
  operational tiers (<M4 ignore, M4–5 monitor, M5–6 minor alert,
  M6+ major alert). The cost is asymmetric: false M6+ alerts have
  economic cost; missed M6+ alerts cost lives.

### 6. Statistical hygiene

- **Bootstrap 95% CIs** on every reported number. Seismology reviewers
  will ask.
- **Paired significance tests** when comparing models (same test events,
  different predictions — paired bootstrap or Wilcoxon signed-rank).
- **Per-event vs. per-station aggregation**: large events with many
  recording stations bias averages. Report both "mean over stations"
  and "mean over events (averaged per-event first)."

### `results.csv` schema

One row per run. Columns:

```
exp_id, timestamp, git_sha, config_hash, seed, rung, model,
train_region, test_region, split_strategy, n_train, n_test,
mae, rmse, bias,
acc_0p3, acc_0p5, acc_1p0,
mae_m3, mae_m4, mae_m5, mae_m6plus,
bias_m6plus,
probe_r2_fc, probe_r2_stressdrop, probe_r2_duration,
mmd_source_target,
notes
```

Add columns as needed; never remove them. Bootstrap CIs, few-shot
curves, and time-to-estimate curves are logged per-run under
`experiments/<exp_id>/` rather than flattened into the CSV.

## Repo layout

```
earthquake-xfer/
├── CLAUDE.md                    # this file
├── README.md
├── pyproject.toml
├── configs/
│   ├── baseline_pd.yaml
│   ├── baseline_cnn.yaml
│   └── split_universal.yaml
├── data/
│   ├── raw/                     # read-only
│   │   ├── stead/
│   │   ├── vs30/
│   │   └── crust1/
│   └── processed/               # cached features, splits
├── src/eqxfer/
│   ├── data/
│   │   ├── stead_loader.py
│   │   ├── geological.py        # VS30 + CRUST1 interpolators
│   │   ├── splits.py            # geographic splits + tests
│   │   └── filters.py           # SNR, P-arrival, magnitude
│   ├── features/
│   │   ├── pd_tauc.py           # classical EEW features
│   │   └── waveform.py          # bandpass, detrend
│   ├── models/
│   │   ├── pd_linear.py         # rung 1
│   │   ├── raw_cnn.py           # rung 3
│   │   ├── physics_cnn.py       # rung 4
│   │   └── split_transfer.py    # rung 5 (the hypothesis)
│   ├── training/
│   │   ├── loops.py
│   │   ├── losses.py
│   │   └── schedulers.py
│   └── evaluation/
│       ├── metrics.py           # MAE/RMSE/bias, ±0.3/0.5/1.0, mag-binned, bootstrap CIs
│       ├── embedding_probes.py  # corner freq regression from embeds
│       ├── transfer_eval.py     # zero-shot + few-shot protocols
│       └── alignment.py         # source/target embedding overlap
├── tests/
│   ├── test_splits_no_leakage.py
│   ├── test_features_reproducible.py
│   └── test_physics_branch_has_no_regional_input.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_ablation.py
└── experiments/
    ├── results.csv              # one row per run
    └── <exp_id>/                # config + logs + checkpoints per run
```

## Target regions and why

| Region | Tectonic regime | STEAD traces | Why it matters |
|---|---|---:|---|
| California | Strike-slip (San Andreas) | 602k | Source domain, dense network, abundant data |
| Greece | Hellenic subduction + normal | 21k | Moderate difficulty, different mechanism, workable data |
| Chile | Megathrust subduction | 5.1k | Hardest — maximally different from California; spans M2.4–M7.7 |

**Excluded**: Japan (zero STEAD coverage — would require separate NIED
K-NET/KiK-net pull), Turkey (only ~1.2k STEAD traces, statistically
underpowered for transfer evaluation). Both are candidate future-work
extensions but out of scope for the current paper.

If Chile transfer works at all, the method is real. Greece is the
sanity-check middle case — different tectonics from California but not
a full megathrust jump, so moderate transfer difficulty expected.

## Style and tooling

- Python 3.11+, type hints everywhere, mypy in CI.
- Dataclasses or Pydantic models for configs. No free-floating dicts.
- Pure functions in `features/`. No hidden state, no global caches.
- One architecture per file under `models/`. Easy to diff, easy to ablate.
- `pytest` for tests, `ruff` for linting, `black` for formatting.
- Log with `structlog` or similar — every run produces a JSON log.
- PyTorch Lightning optional but welcome; we care about research speed
  over framework purity.
- Weights & Biases or MLflow for experiment tracking in addition to the
  results.csv.

## Definition of done for this project

### Per-rung pass/fail gates

Each rung must land within its target ±0.3 accuracy range with
bootstrap 95% CI overlapping the target midpoint, AND beat the prior
rung by non-overlapping CIs. If either fails, stop and debug before
climbing further.

| Rung | Model | Target ±0.3 acc (California in-region) |
|---|---|---:|
| 1 | Pd log-linear | 55–65% |
| 2 | Pd + τc linear | 60–70% |
| 3 | Raw-waveform CNN | 65–75% |
| 4 | CNN + 5 physics features | 68–78% |
| 5 | Split architecture | Rung 4 + ≥ 3% |

Targets for rungs 1–4 are derived from EEW literature (Wu & Kanamori
2005; Allen & Kanamori 2003). Treat them as the outer envelope — the
lower bound is "expected floor," the upper bound is "expected ceiling."

### Paper-level result

A paper-quality result looks like:
- Rung 5 beats rung 4 in-region by ≥ 3% ±0.3 accuracy.
- Zero-shot transfer to Greece: within 10% of in-region performance
  on ±0.3 accuracy.
- Zero-shot transfer to Chile: beats the magnitude-mean baseline
  substantially, and beats a from-scratch model trained on <500 Chile
  events.
- Embedding probes show the universal encoder captures corner frequency
  and rupture duration with R² > 0.5.
- Source and target embeddings overlap in UMAP (visual + MMD metric).
- Ablation table shows each architectural component contributes.

If we don't hit these, we write up what we found and why the hypothesis
didn't fully hold. That's also a real result.

Ensure my laptop gpu (Nvidia) is being used at all time during the training. 