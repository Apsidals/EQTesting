"""Metric panel + bootstrap CIs. Same function serves every rung so the
numbers are comparable across the ladder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

MAG_BIN_EDGES = [(3.0, 4.0, "m3"), (4.0, 5.0, "m4"), (5.0, 6.0, "m5"), (6.0, 99.0, "m6plus")]


def _acc_within(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    return float((np.abs(y_pred - y_true) <= tol).mean())


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Returns nan if y_true has zero variance
    (e.g. an empty slice or a single-magnitude bin)."""
    if len(y_true) < 2:
        return float("nan")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


@dataclass
class MetricPanel:
    n: int
    mae: float
    rmse: float
    r2: float
    bias: float
    acc_0p3: float
    acc_0p5: float
    acc_1p0: float
    mae_m3: float
    mae_m4: float
    mae_m5: float
    mae_m6plus: float
    rmse_m3: float
    rmse_m4: float
    rmse_m5: float
    rmse_m6plus: float
    n_m3: int
    n_m4: int
    n_m5: int
    n_m6plus: int
    bias_m6plus: float
    mae_ci95: tuple[float, float] = (float("nan"), float("nan"))
    acc_0p3_ci95: tuple[float, float] = (float("nan"), float("nan"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mae_ci95_lo"], d["mae_ci95_hi"] = d.pop("mae_ci95")
        d["acc_0p3_ci95_lo"], d["acc_0p3_ci95_hi"] = d.pop("acc_0p3_ci95")
        return d


def _binned(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lo, hi, name in MAG_BIN_EDGES:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            out[name] = {
                "mae": _mae(y_true[mask], y_pred[mask]),
                "rmse": _rmse(y_true[mask], y_pred[mask]),
                "n": int(mask.sum()),
            }
        else:
            out[name] = {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return out


def compute_metric_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> MetricPanel:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    binned = _binned(y_true, y_pred)
    m6plus_mask = y_true >= 6.0
    bias_m6plus = _bias(y_true[m6plus_mask], y_pred[m6plus_mask]) if m6plus_mask.any() else float("nan")

    panel = MetricPanel(
        n=int(len(y_true)),
        mae=_mae(y_true, y_pred),
        rmse=_rmse(y_true, y_pred),
        r2=_r2(y_true, y_pred),
        bias=_bias(y_true, y_pred),
        acc_0p3=_acc_within(y_true, y_pred, 0.3),
        acc_0p5=_acc_within(y_true, y_pred, 0.5),
        acc_1p0=_acc_within(y_true, y_pred, 1.0),
        mae_m3=binned["m3"]["mae"],
        mae_m4=binned["m4"]["mae"],
        mae_m5=binned["m5"]["mae"],
        mae_m6plus=binned["m6plus"]["mae"],
        rmse_m3=binned["m3"]["rmse"],
        rmse_m4=binned["m4"]["rmse"],
        rmse_m5=binned["m5"]["rmse"],
        rmse_m6plus=binned["m6plus"]["rmse"],
        n_m3=binned["m3"]["n"],
        n_m4=binned["m4"]["n"],
        n_m5=binned["m5"]["n"],
        n_m6plus=int(m6plus_mask.sum()),
        bias_m6plus=bias_m6plus,
    )

    if bootstrap_samples > 0 and len(y_true) > 1:
        rng = np.random.default_rng(seed)
        n = len(y_true)
        mae_samples = np.empty(bootstrap_samples)
        acc_samples = np.empty(bootstrap_samples)
        for i in range(bootstrap_samples):
            idx = rng.integers(0, n, size=n)
            mae_samples[i] = _mae(y_true[idx], y_pred[idx])
            acc_samples[i] = _acc_within(y_true[idx], y_pred[idx], 0.3)
        panel.mae_ci95 = (float(np.quantile(mae_samples, 0.025)), float(np.quantile(mae_samples, 0.975)))
        panel.acc_0p3_ci95 = (float(np.quantile(acc_samples, 0.025)), float(np.quantile(acc_samples, 0.975)))

    return panel


def format_panel(panel: MetricPanel) -> str:
    lines = [
        f"n = {panel.n:,} (n_m6plus = {panel.n_m6plus})",
        f"MAE       = {panel.mae:.4f}  95% CI [{panel.mae_ci95[0]:.4f}, {panel.mae_ci95[1]:.4f}]",
        f"RMSE      = {panel.rmse:.4f}",
        f"R²        = {panel.r2:+.4f}",
        f"Bias      = {panel.bias:+.4f}",
        f"Acc ±0.3  = {panel.acc_0p3:.4f}  95% CI [{panel.acc_0p3_ci95[0]:.4f}, {panel.acc_0p3_ci95[1]:.4f}]",
        f"Acc ±0.5  = {panel.acc_0p5:.4f}",
        f"Acc ±1.0  = {panel.acc_1p0:.4f}",
        f"M[3–4]  MAE={panel.mae_m3:.4f}  RMSE={panel.rmse_m3:.4f}  (n={panel.n_m3})",
        f"M[4–5]  MAE={panel.mae_m4:.4f}  RMSE={panel.rmse_m4:.4f}  (n={panel.n_m4})",
        f"M[5–6]  MAE={panel.mae_m5:.4f}  RMSE={panel.rmse_m5:.4f}  (n={panel.n_m5})",
        f"M[6+]   MAE={panel.mae_m6plus:.4f}  RMSE={panel.rmse_m6plus:.4f}  "
        f"(n={panel.n_m6plus}, bias {panel.bias_m6plus:+.4f})",
    ]
    return "\n".join(lines)


def format_panel_compact(panel: MetricPanel) -> str:
    """Two-line epoch-friendly summary: overall metrics + per-magnitude-bin
    MAE/RMSE. ASCII only so Windows consoles don't mangle it."""
    line_main = (
        f"MAE={panel.mae:.4f}  RMSE={panel.rmse:.4f}  "
        f"R2={panel.r2:+.4f}  bias={panel.bias:+.4f}  "
        f"acc +/-0.3/0.5/1.0 = {panel.acc_0p3:.3f}/{panel.acc_0p5:.3f}/{panel.acc_1p0:.3f}"
    )

    def _bin(label: str, mae: float, rmse: float, n: int) -> str:
        if n == 0:
            return f"{label} n=0"
        return f"{label} MAE/RMSE={mae:.3f}/{rmse:.3f} (n={n})"

    line_bins = "  |  ".join(
        [
            _bin("M[3-4]", panel.mae_m3, panel.rmse_m3, panel.n_m3),
            _bin("M[4-5]", panel.mae_m4, panel.rmse_m4, panel.n_m4),
            _bin("M[5-6]", panel.mae_m5, panel.rmse_m5, panel.n_m5),
            _bin("M[6+]", panel.mae_m6plus, panel.rmse_m6plus, panel.n_m6plus),
        ]
    )
    return line_main + "\n         " + line_bins
