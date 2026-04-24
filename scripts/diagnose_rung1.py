"""Quick diagnostics on rung 1 behavior. Answers:
- What's the actual magnitude distribution in train/val/test?
- What do the bin weights look like?
- How does the fit change if we drop the weighting entirely?
- How does it change if we drop the cap to 5x?
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eqxfer.config import load_run_config  # noqa: E402
from eqxfer.data.filters import load_or_compute_features  # noqa: E402
from eqxfer.data.splits import make_splits  # noqa: E402
from eqxfer.data.stead_loader import SteadLoader  # noqa: E402
from eqxfer.evaluation.metrics import compute_metric_panel, format_panel  # noqa: E402
from eqxfer.models.pd_linear import PdLinear, bin_balanced_weights  # noqa: E402


def _mag_hist(mags: np.ndarray, label: str) -> None:
    bins = Counter(int(np.floor(m)) for m in mags)
    print(f"\n{label} (n={len(mags)}):")
    for k in sorted(bins):
        pct = 100 * bins[k] / len(mags)
        print(f"  M[{k}-{k+1}): {bins[k]:>6,}  ({pct:5.2f}%)")
    print(f"  min={mags.min():.2f}  median={np.median(mags):.2f}  "
          f"p95={np.percentile(mags, 95):.2f}  max={mags.max():.2f}")


def _fit_and_report(pd_train, mag_train, pd_test, mag_test, weights, name: str) -> None:
    model = PdLinear.fit(pd_train, mag_train, weights=weights)
    y_pred = model.predict(pd_test)
    panel = compute_metric_panel(mag_test, y_pred, bootstrap_samples=500)
    print(f"\n=== {name} ===")
    print(f"Fitted: M = {model.intercept:+.4f} + {model.slope:+.4f} * log10(Pd)")
    print(format_panel(panel))


def main() -> None:
    cfg = load_run_config(Path("configs/baseline_pd.yaml"))
    loader = SteadLoader(cfg.loader)
    features = load_or_compute_features(loader)
    split = make_splits(loader.metadata, cfg.split)

    meta = loader.metadata.set_index("trace_name")

    def xy(names):
        return (
            features.loc[names, "pd_z"].to_numpy(),
            meta.loc[names, "source_magnitude"].to_numpy(),
        )

    pd_train, mag_train = xy(split.train)
    pd_val, mag_val = xy(split.val)
    pd_test, mag_test = xy(split.test)

    print("-" * 60)
    print("Magnitude distributions")
    print("-" * 60)
    _mag_hist(mag_train, "TRAIN")
    _mag_hist(mag_val, "VAL")
    _mag_hist(mag_test, "TEST (Ridgecrest/Coso)")

    print("\n" + "-" * 60)
    print("Weight distribution (cap=20)")
    print("-" * 60)
    w20 = bin_balanced_weights(mag_train, bin_width=1.0, cap=20.0)
    for lo in range(3, 7):
        mask = (mag_train >= lo) & (mag_train < lo + 1)
        if mask.any():
            print(f"  M[{lo}-{lo+1}): n={mask.sum():>6}  weight={w20[mask][0]:.2f}  "
                  f"total_signal={w20[mask].sum():.0f}")

    print("\n" + "-" * 60)
    print("Fit variants: how much of the bias is from weighting?")
    print("-" * 60)
    _fit_and_report(pd_train, mag_train, pd_test, mag_test, None, "UNWEIGHTED")
    _fit_and_report(
        pd_train, mag_train, pd_test, mag_test,
        bin_balanced_weights(mag_train, bin_width=1.0, cap=5.0),
        "WEIGHTED cap=5",
    )
    _fit_and_report(
        pd_train, mag_train, pd_test, mag_test,
        bin_balanced_weights(mag_train, bin_width=1.0, cap=20.0),
        "WEIGHTED cap=20 (current)",
    )


if __name__ == "__main__":
    main()
