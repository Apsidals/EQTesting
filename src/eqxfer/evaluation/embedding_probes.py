"""Physics probes for the Universal Physics Encoder.

The claim "the universal encoder learned physics" is only meaningful if
physics-interpretable quantities can be regressed out of its frozen
embeddings with reasonable R². Per CLAUDE.md section "Physics-probe
metrics", targets are:

- Corner frequency (fc_z) — spectral corner, proxy for stress drop + size.
- Rupture duration (tau_c) — Nakamura τc on Z channel.
- Stress-drop proxy: log10(Pd * fc^3), Brune-model scaling.

If any of these comes back with R² < 0.3, the hypothesis that the encoder
is learning universal source physics is in trouble. That's a publishable
negative result; report it honestly.

Probe is ridge regression (linear, closed form) — we specifically want a
simple decoder so strong probe R² implies the structure is in the
embedding, not in a deep probe head.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class ProbeResult:
    target: str
    n_train: int
    n_test: int
    r2: float
    rmse: float

    def to_dict(self) -> dict:
        return asdict(self)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def probe(
    embeddings_train: np.ndarray,
    targets_train: np.ndarray,
    embeddings_test: np.ndarray,
    targets_test: np.ndarray,
    target_name: str,
    alpha: float = 1.0,
) -> ProbeResult:
    """Ridge regression from embeddings → target. Returns R² on held-out."""
    mask_tr = np.isfinite(targets_train)
    mask_te = np.isfinite(targets_test)
    model = Ridge(alpha=alpha)
    model.fit(embeddings_train[mask_tr], targets_train[mask_tr])
    y_pred = model.predict(embeddings_test[mask_te])
    y_true = targets_test[mask_te]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return ProbeResult(
        target=target_name,
        n_train=int(mask_tr.sum()),
        n_test=int(mask_te.sum()),
        r2=_r2(y_true, y_pred),
        rmse=rmse,
    )


def probe_physics_battery(
    embeddings_train: np.ndarray,
    embeddings_test: np.ndarray,
    features_train,  # DataFrame with pd_z, tau_c, fc_z
    features_test,
) -> dict[str, ProbeResult]:
    """Run the three canonical probes. Features are expected to be the
    physics-features DataFrame with rows aligned to the embedding rows."""
    out: dict[str, ProbeResult] = {}

    # Log10 everything strictly positive — much easier for a linear probe.
    fc_train = np.log10(features_train["fc_z"].to_numpy(dtype=np.float64))
    fc_test = np.log10(features_test["fc_z"].to_numpy(dtype=np.float64))
    tauc_train = np.log10(features_train["tau_c"].to_numpy(dtype=np.float64))
    tauc_test = np.log10(features_test["tau_c"].to_numpy(dtype=np.float64))

    stress_train = (
        np.log10(features_train["pd_z"].to_numpy(dtype=np.float64))
        + 3.0 * fc_train
    )
    stress_test = (
        np.log10(features_test["pd_z"].to_numpy(dtype=np.float64))
        + 3.0 * fc_test
    )

    out["corner_frequency"] = probe(
        embeddings_train, fc_train, embeddings_test, fc_test,
        target_name="log10_fc",
    )
    out["rupture_duration"] = probe(
        embeddings_train, tauc_train, embeddings_test, tauc_test,
        target_name="log10_tau_c",
    )
    out["stress_drop_proxy"] = probe(
        embeddings_train, stress_train, embeddings_test, stress_test,
        target_name="log10_pd_plus_3log10_fc",
    )
    return out
