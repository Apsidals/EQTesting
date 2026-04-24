"""Entry point for training a rung.

Usage:
    python scripts/train.py --config configs/baseline_pd.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eqxfer.config import RunConfig, config_hash, load_run_config  # noqa: E402
from eqxfer.data.dataset import WaveformDataset  # noqa: E402
from eqxfer.data.samplers import EventGroupedBatchSampler  # noqa: E402
from eqxfer.data.waveform_cache import WaveformCache  # noqa: E402
from eqxfer.data.filters import (  # noqa: E402
    load_or_compute_features,
    load_or_compute_site_features,
)
from eqxfer.data.splits import SplitResult, make_splits  # noqa: E402
from eqxfer.data.stead_loader import SteadLoader  # noqa: E402
from eqxfer.evaluation.embedding_probes import probe_physics_battery  # noqa: E402
from eqxfer.evaluation.logger import (  # noqa: E402
    append_result_row,
    get_git_sha,
    make_exp_dir,
    make_exp_id,
)
from eqxfer.evaluation.metrics import compute_metric_panel, format_panel  # noqa: E402
from eqxfer.evaluation.transfer_eval import build_loader  # noqa: E402
from eqxfer.models.pd_linear import PdLinear, bin_balanced_weights  # noqa: E402
from eqxfer.models.split_transfer import SplitTransferModel  # noqa: E402
from eqxfer.training.loops import (  # noqa: E402
    extract_physics_embeddings,
    predict,
    select_device,
    train_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a baseline rung.")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _gather_xy(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    trace_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    idx = metadata.set_index("trace_name").loc[trace_names]
    pd_values = features.loc[trace_names, "pd_z"].to_numpy()
    magnitudes = idx["source_magnitude"].to_numpy()
    return pd_values, magnitudes


def train_rung1(cfg: RunConfig) -> None:
    exp_id = make_exp_id(cfg.rung)
    exp_dir = make_exp_dir(exp_id)
    print(f"\n=== Rung {cfg.rung}: {cfg.model_name} ({exp_id}) ===\n", flush=True)

    loader = SteadLoader(cfg.loader)
    features = load_or_compute_features(loader)

    split = make_splits(loader.metadata, cfg.split)
    sizes = split.sizes()
    print(f"Splits: train={sizes['train']:,}  val={sizes['val']:,}  test={sizes['test']:,}")

    pd_train, mag_train = _gather_xy(loader.metadata, features, split.train)
    pd_val, mag_val = _gather_xy(loader.metadata, features, split.val)
    pd_test, mag_test = _gather_xy(loader.metadata, features, split.test)

    weights = bin_balanced_weights(mag_train, bin_width=1.0, cap=20.0)
    model = PdLinear.fit(pd_train, mag_train, weights=weights)
    print(f"\nFitted: M = {model.intercept:+.4f} + {model.slope:+.4f} * log10(Pd)")

    y_pred_val = model.predict(pd_val)
    y_pred_test = model.predict(pd_test)

    val_panel = compute_metric_panel(mag_val, y_pred_val)
    test_panel = compute_metric_panel(mag_test, y_pred_test)

    print("\n--- Val ---")
    print(format_panel(val_panel))
    print("\n--- Test (Ridgecrest/Coso holdout) ---")
    print(format_panel(test_panel))

    _write_artifacts(
        exp_dir=exp_dir,
        cfg=cfg,
        model={"intercept": model.intercept, "slope": model.slope},
        val_panel=val_panel,
        test_panel=test_panel,
        split_sizes=sizes,
        predictions=_predictions_df(split.test, mag_test, y_pred_test),
    )

    cfg_hash = config_hash(cfg)
    row = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": get_git_sha(),
        "config_hash": cfg_hash,
        "seed": cfg.split.seed,
        "rung": cfg.rung,
        "model": cfg.model_name,
        "train_region": "California",
        "test_region": "California-Ridgecrest",
        "split_strategy": cfg.split.strategy,
        "n_train": sizes["train"],
        "n_test": sizes["test"],
        "mae": f"{test_panel.mae:.4f}",
        "rmse": f"{test_panel.rmse:.4f}",
        "bias": f"{test_panel.bias:.4f}",
        "acc_0p3": f"{test_panel.acc_0p3:.4f}",
        "acc_0p5": f"{test_panel.acc_0p5:.4f}",
        "acc_1p0": f"{test_panel.acc_1p0:.4f}",
        "mae_m3": f"{test_panel.mae_m3:.4f}",
        "mae_m4": f"{test_panel.mae_m4:.4f}",
        "mae_m5": f"{test_panel.mae_m5:.4f}",
        "mae_m6plus": f"{test_panel.mae_m6plus:.4f}",
        "bias_m6plus": f"{test_panel.bias_m6plus:.4f}",
        "notes": cfg.notes,
    }
    append_result_row(row)
    print(f"\nAppended row to experiments/results.csv (exp_id={exp_id})")


def _predictions_df(
    trace_names: list[str], y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {"trace_name": trace_names, "y_true": y_true, "y_pred": y_pred, "residual": y_pred - y_true}
    )


def _write_artifacts(
    exp_dir: Path,
    cfg: RunConfig,
    model: dict,
    val_panel,
    test_panel,
    split_sizes: dict,
    predictions: pd.DataFrame,
) -> None:
    (exp_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))
    (exp_dir / "model.json").write_text(json.dumps(model, indent=2))
    (exp_dir / "metrics.json").write_text(
        json.dumps(
            {"val": val_panel.to_dict(), "test": test_panel.to_dict(), "split_sizes": split_sizes},
            indent=2,
        )
    )
    predictions.to_csv(exp_dir / "predictions_test.csv", index=False)


def train_rung5(cfg: RunConfig) -> None:
    """Rung 5: split architecture. Pretrain on cfg.loader.region (California)
    with event-grouped train/val + geographic holdout test. Saves a state
    checkpoint consumable by scripts/transfer.py."""
    import torch

    exp_id = make_exp_id(cfg.rung)
    exp_dir = make_exp_dir(exp_id)
    print(f"\n=== Rung {cfg.rung}: {cfg.model_name} ({exp_id}) ===\n", flush=True)

    device = select_device(require_cuda=cfg.train.require_cuda)
    torch.manual_seed(cfg.train.seed)

    loader = SteadLoader(cfg.loader)
    phys_features = load_or_compute_features(loader)
    site_features = load_or_compute_site_features(loader)

    split = make_splits(loader.metadata, cfg.split)
    sizes = split.sizes()
    print(f"Splits: train={sizes['train']:,}  val={sizes['val']:,}  test={sizes['test']:,}")

    # Preprocessed-waveform cache: one memmapped .npy over the union of
    # train/val/test, keyed by preprocessing config + trace set. Built once,
    # then every epoch hits a numpy slice instead of HDF5 read + filtfilt.
    # Defaults to ~/.cache/eqxfer/waveforms (WSL native disk) so it doesn't
    # inherit the /mnt/c/ 9P bridge penalty.
    cache_trace_names = loader.metadata["trace_name"].tolist()
    cache_p_samples = loader.metadata["p_arrival_sample"].astype(int).to_numpy()
    cache = WaveformCache(cfg.loader, trace_names=cache_trace_names)
    print(f"[cache] dir={cache.cache_dir}  key={cache.key}", flush=True)
    if cache.exists():
        print(f"[cache] hit — {cache.data_path.name}", flush=True)
    else:
        cache.build(cache_p_samples)

    train_ds = WaveformDataset(
        cfg.loader,
        loader.metadata,
        site_features,
        split.train,
        phys_features=phys_features,
        waveform_cache=cache,
    )
    val_ds = WaveformDataset(
        cfg.loader,
        loader.metadata,
        site_features,
        split.val,
        waveform_cache=cache,
    )
    test_ds = WaveformDataset(
        cfg.loader,
        loader.metadata,
        site_features,
        split.test,
        waveform_cache=cache,
    )

    # Event-grouped batch sampler — replaces the old per-trace
    # WeightedRandomSampler. Each batch is events_per_batch blocks of
    # stations_per_event traces from the same source event; the training
    # loop reshapes to (E, K, D) to compute L_sep (within-event variance
    # of the physics embedding). Magnitude balancing now operates at the
    # event level, still capped by train.sampler_cap.
    train_meta_idx = loader.metadata.set_index("trace_name").loc[split.train]
    event_ids_train = train_meta_idx["source_id"].to_numpy()
    magnitudes_train = train_meta_idx["source_magnitude"].astype(np.float32).to_numpy()
    grouped_sampler = EventGroupedBatchSampler(
        event_ids=event_ids_train,
        magnitudes=magnitudes_train,
        events_per_batch=cfg.separation.events_per_batch,
        stations_per_event=cfg.separation.stations_per_event,
        min_stations_per_event=cfg.separation.min_stations_per_event,
        bin_width=cfg.train.bin_width,
        sampler_cap=cfg.train.sampler_cap,
        seed=cfg.train.seed,
    )
    n_eligible = grouped_sampler.n_eligible_traces()
    n_total = len(split.train)
    print(
        f"[sampler] event-grouped: "
        f"{len(grouped_sampler.eligible_events):,} events (>= "
        f"{cfg.separation.min_stations_per_event} stations), "
        f"{n_eligible:,}/{n_total:,} traces reachable "
        f"({100*n_eligible/max(1,n_total):.1f}%). "
        f"{len(grouped_sampler):,} batches/epoch of "
        f"{grouped_sampler.batch_size} samples "
        f"({cfg.separation.events_per_batch} events × "
        f"{cfg.separation.stations_per_event} stations).",
        flush=True,
    )
    train_loader = build_loader(
        train_ds,
        cfg.train.batch_size,
        cfg.train.num_workers,
        batch_sampler=grouped_sampler,
        prefetch_factor=cfg.train.prefetch_factor,
    )
    val_loader = build_loader(
        val_ds,
        cfg.train.batch_size,
        cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
    )
    test_loader = build_loader(
        test_ds,
        cfg.train.batch_size,
        cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
    )

    model = SplitTransferModel(cfg.model_hparams)
    print(
        f"Model params: "
        f"universal={sum(p.numel() for p in model.universal.parameters()):,}  "
        f"site={sum(p.numel() for p in model.site.parameters()):,}  "
        f"fusion={sum(p.numel() for p in model.fusion.parameters()):,}",
        flush=True,
    )

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg.train,
        device=device,
        exp_dir=exp_dir,
        tag="pretrain",
        aux_cfg=cfg.aux_losses,
        sep_cfg=cfg.separation,
    )

    y_val_true, y_val_pred = predict(model, val_loader, device)
    y_test_true, y_test_pred = predict(model, test_loader, device)
    val_panel = compute_metric_panel(y_val_true, y_val_pred)
    test_panel = compute_metric_panel(y_test_true, y_test_pred)

    print("\n--- Val ---\n" + format_panel(val_panel))
    print("\n--- Test (Ridgecrest/Coso holdout) ---\n" + format_panel(test_panel))

    # Physics probes: regress fc / tau_c / stress-drop proxy from frozen
    # universal-encoder embeddings. Targets our rule-of-thumb R² > 0.5 for
    # the hypothesis to hold.
    val_embed = extract_physics_embeddings(model, val_loader, device)
    test_embed = extract_physics_embeddings(model, test_loader, device)
    probes = probe_physics_battery(
        embeddings_train=val_embed,
        embeddings_test=test_embed,
        features_train=phys_features.loc[split.val],
        features_test=phys_features.loc[split.test],
    )
    probe_r2 = {k: float(p.r2) for k, p in probes.items()}
    print("\n--- Physics probes ---")
    for name, pr in probes.items():
        print(f"  {name}: R²={pr.r2:.3f}  RMSE={pr.rmse:.3f}  (n_test={pr.n_test})")

    (exp_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
    )
    (exp_dir / "metrics.json").write_text(
        json.dumps(
            {
                "val": val_panel.to_dict(),
                "test": test_panel.to_dict(),
                "split_sizes": sizes,
                "best_epoch": result.best_epoch,
                "best_val_mae": result.best_val_mae,
                "probes": {k: p.to_dict() for k, p in probes.items()},
            },
            indent=2,
        )
    )
    pd.DataFrame(
        {
            "trace_name": split.test,
            "y_true": y_test_true,
            "y_pred": y_test_pred,
            "residual": y_test_pred - y_test_true,
        }
    ).to_csv(exp_dir / "predictions_test.csv", index=False)

    cfg_hash = config_hash(cfg)
    row = {
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": get_git_sha(),
        "config_hash": cfg_hash,
        "seed": cfg.split.seed,
        "rung": cfg.rung,
        "model": cfg.model_name,
        "train_region": cfg.loader.region,
        "test_region": f"{cfg.loader.region}-Ridgecrest" if cfg.split.strategy == "california_ridgecrest" else cfg.loader.region,
        "split_strategy": cfg.split.strategy,
        "n_train": sizes["train"],
        "n_test": sizes["test"],
        "mae": f"{test_panel.mae:.4f}",
        "rmse": f"{test_panel.rmse:.4f}",
        "bias": f"{test_panel.bias:.4f}",
        "acc_0p3": f"{test_panel.acc_0p3:.4f}",
        "acc_0p5": f"{test_panel.acc_0p5:.4f}",
        "acc_1p0": f"{test_panel.acc_1p0:.4f}",
        "mae_m3": f"{test_panel.mae_m3:.4f}",
        "mae_m4": f"{test_panel.mae_m4:.4f}",
        "mae_m5": f"{test_panel.mae_m5:.4f}",
        "mae_m6plus": f"{test_panel.mae_m6plus:.4f}",
        "bias_m6plus": f"{test_panel.bias_m6plus:.4f}",
        "probe_r2_fc": f"{probe_r2.get('corner_frequency', float('nan')):.4f}",
        "probe_r2_stressdrop": f"{probe_r2.get('stress_drop_proxy', float('nan')):.4f}",
        "probe_r2_duration": f"{probe_r2.get('rupture_duration', float('nan')):.4f}",
        "notes": cfg.notes,
    }
    append_result_row(row)
    print(
        f"\nCheckpoint: {result.best_state_path}\n"
        f"Row appended to experiments/results.csv (exp_id={exp_id})"
    )


def main() -> None:
    args = parse_args()
    cfg = load_run_config(args.config)
    if cfg.rung == 1:
        train_rung1(cfg)
    elif cfg.rung == 5:
        train_rung5(cfg)
    else:
        raise NotImplementedError(f"Rung {cfg.rung} not yet implemented")


if __name__ == "__main__":
    main()
