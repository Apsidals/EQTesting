# Earthquake Cross-Tectonic Magnitude Transfer (`eqxfer`)

Testing whether earthquake **source physics** is universal and **site response**
is regional, by forcing that separation into the model architecture and
measuring how well the physics branch transfers across tectonic regimes.

See `CLAUDE.md` for the full research spec, hard rules, and baseline ladder.

## Layout

- `configs/` — YAML run configs, one per experiment class
- `src/eqxfer/` — library code (data, features, models, training, evaluation)
- `tests/` — leakage tests, reproducibility tests, architectural invariants
- `scripts/` — entry points: `train.py`, `evaluate.py`, `run_ablation.py`
- `experiments/` — `results.csv` + per-run artifacts
- `data/raw/` — read-only inputs (STEAD, VS30, CRUST1)
- `data/processed/` — cached features and splits

## Install (dev)

```bash
pip install -e ".[dev,tracking,viz]"
```

## Baseline ladder

Run in order. Do not skip rungs.

1. `scripts/train.py --config configs/baseline_pd.yaml`       # Pd log-linear
2. Pd + τc linear                                              # two features
3. `scripts/train.py --config configs/baseline_cnn.yaml`      # raw-waveform CNN
4. CNN + 3–5 curated physics features
5. `scripts/train.py --config configs/split_universal.yaml`   # the hypothesis
6. Transfer experiments (only after rung 5 works in-region)
