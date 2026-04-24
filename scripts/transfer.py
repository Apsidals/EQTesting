"""Transfer experiment orchestrator.

Usage:
    python scripts/transfer.py \
        --config configs/transfer_greece.yaml \
        --pretrained-state experiments/rung5_.../pretrain_best.pt \
        [--source-config configs/split_universal.yaml]  # for MMD / alignment

The pretrained-state is produced by scripts/train.py on a rung-5 config.
If --source-config is given, a small sample of California embeddings is
pulled to compute source↔target MMD and silhouette on the universal
encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from eqxfer.config import RunConfig, config_hash, load_run_config  # noqa: E402
from eqxfer.data.dataset import WaveformDataset  # noqa: E402
from eqxfer.data.waveform_cache import WaveformCache  # noqa: E402
from eqxfer.data.filters import (  # noqa: E402
    load_or_compute_features,
    load_or_compute_site_features,
)
from eqxfer.data.splits import make_splits  # noqa: E402
from eqxfer.data.stead_loader import SteadLoader  # noqa: E402
from eqxfer.evaluation.alignment import rbf_mmd2, region_silhouette  # noqa: E402
from eqxfer.evaluation.embedding_probes import probe_physics_battery  # noqa: E402
from eqxfer.evaluation.logger import (  # noqa: E402
    append_result_row,
    get_git_sha,
    make_exp_dir,
    make_exp_id,
)
from eqxfer.evaluation.metrics import format_panel  # noqa: E402
from eqxfer.evaluation.transfer_eval import (  # noqa: E402
    TransferReport,
    build_loader,
    few_shot_finetune,
    from_scratch_baseline,
    zero_shot_evaluate,
)
from eqxfer.models.split_transfer import SplitTransferModel  # noqa: E402
from eqxfer.training.loops import extract_physics_embeddings, select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a transfer-learning experiment.")
    p.add_argument("--config", type=Path, required=True, help="Target-region config yaml.")
    p.add_argument(
        "--pretrained-state",
        type=Path,
        required=True,
        help="Path to the pretrained SplitTransferModel state_dict (from scripts/train.py).",
    )
    p.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Optional source-region config (California) for MMD / alignment metrics.",
    )
    p.add_argument(
        "--source-embed-sample",
        type=int,
        default=2000,
        help="Sample this many source traces for embedding alignment (MMD, silhouette).",
    )
    return p.parse_args()


def _row_for_stage(
    cfg: RunConfig, exp_id: str, stage_result, mmd: float | None, probe_r2: dict
) -> dict:
    panel = stage_result.panel
    return {
        "exp_id": f"{exp_id}_{stage_result.stage}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": get_git_sha(),
        "config_hash": config_hash(cfg),
        "seed": cfg.split.seed,
        "rung": cfg.rung,
        "model": f"{cfg.model_name}:{stage_result.stage}",
        "train_region": "California (pretrain)",
        "test_region": cfg.loader.region,
        "split_strategy": cfg.split.strategy,
        "n_train": stage_result.n_adapt_events,
        "n_test": panel.n,
        "mae": f"{panel.mae:.4f}",
        "rmse": f"{panel.rmse:.4f}",
        "bias": f"{panel.bias:.4f}",
        "acc_0p3": f"{panel.acc_0p3:.4f}",
        "acc_0p5": f"{panel.acc_0p5:.4f}",
        "acc_1p0": f"{panel.acc_1p0:.4f}",
        "mae_m3": f"{panel.mae_m3:.4f}",
        "mae_m4": f"{panel.mae_m4:.4f}",
        "mae_m5": f"{panel.mae_m5:.4f}",
        "mae_m6plus": f"{panel.mae_m6plus:.4f}",
        "bias_m6plus": f"{panel.bias_m6plus:.4f}",
        "probe_r2_fc": f"{probe_r2.get('corner_frequency', float('nan')):.4f}",
        "probe_r2_stressdrop": f"{probe_r2.get('stress_drop_proxy', float('nan')):.4f}",
        "probe_r2_duration": f"{probe_r2.get('rupture_duration', float('nan')):.4f}",
        "mmd_source_target": f"{mmd:.4f}" if mmd is not None else "",
        "notes": f"{cfg.notes} [stage={stage_result.stage}]",
    }


def main() -> None:
    args = parse_args()
    cfg = load_run_config(args.config)
    if cfg.transfer is None:
        raise ValueError(
            "transfer config is required — populate the `transfer:` block in the yaml"
        )
    if cfg.transfer.target_region != cfg.loader.region:
        raise ValueError(
            f"transfer.target_region ({cfg.transfer.target_region}) must equal "
            f"loader.region ({cfg.loader.region})"
        )

    exp_id = make_exp_id(cfg.rung)
    exp_dir = make_exp_dir(exp_id)
    print(f"\n=== Transfer to {cfg.loader.region} ({exp_id}) ===\n", flush=True)

    device = select_device(require_cuda=cfg.train.require_cuda)
    torch.manual_seed(cfg.train.seed)

    # --- Target-region data pipeline ---
    tgt_loader = SteadLoader(cfg.loader)
    tgt_phys = load_or_compute_features(tgt_loader)
    tgt_site = load_or_compute_site_features(tgt_loader)
    tgt_split = make_splits(tgt_loader.metadata, cfg.split)
    print(
        f"Target splits: train={len(tgt_split.train):,}  val={len(tgt_split.val):,}  "
        f"test={len(tgt_split.test):,}",
        flush=True,
    )

    # Build a target-region waveform cache once, reuse across all dataset
    # constructions (val/test loaders plus each few-shot / from-scratch
    # adapt set). Cache is keyed by (preproc config + trace set), so this
    # Chile cache coexists with the California cache from pretraining —
    # different files, no overwrite.
    tgt_cache_trace_names = tgt_loader.metadata["trace_name"].tolist()
    tgt_cache_p_samples = (
        tgt_loader.metadata["p_arrival_sample"].astype(int).to_numpy()
    )
    tgt_cache = WaveformCache(cfg.loader, trace_names=tgt_cache_trace_names)
    print(
        f"[cache] target dir={tgt_cache.cache_dir}  key={tgt_cache.key}", flush=True
    )
    if tgt_cache.exists():
        print(f"[cache] hit — {tgt_cache.data_path.name}", flush=True)
    else:
        tgt_cache.build(tgt_cache_p_samples)

    # Val / test loaders stay fixed across all stages.
    tgt_val_ds = WaveformDataset(
        cfg.loader, tgt_loader.metadata, tgt_site, tgt_split.val,
        waveform_cache=tgt_cache,
    )
    tgt_test_ds = WaveformDataset(
        cfg.loader, tgt_loader.metadata, tgt_site, tgt_split.test,
        waveform_cache=tgt_cache,
    )
    tgt_val_loader = build_loader(
        tgt_val_ds,
        cfg.train.batch_size,
        cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
    )
    tgt_test_loader = build_loader(
        tgt_test_ds,
        cfg.train.batch_size,
        cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
    )

    # --- Load pretrained source model ---
    pretrained_state = torch.load(args.pretrained_state, map_location=device)
    print(f"Loaded pretrained state from {args.pretrained_state}")

    # --- Stage 0: zero-shot evaluation on target test ---
    model = SplitTransferModel(cfg.model_hparams).to(device)
    model.load_state_dict(pretrained_state)
    zs_panel = zero_shot_evaluate(model, tgt_test_loader, device)
    print(f"\n--- Zero-shot {cfg.loader.region} test ---\n{format_panel(zs_panel)}")

    # --- Target probes: does the frozen universal encoder still expose
    # physics on target-region data? ---
    tgt_test_embed = extract_physics_embeddings(model, tgt_test_loader, device)
    tgt_val_embed = extract_physics_embeddings(model, tgt_val_loader, device)
    probes = probe_physics_battery(
        embeddings_train=tgt_val_embed,
        embeddings_test=tgt_test_embed,
        features_train=tgt_phys.loc[tgt_split.val],
        features_test=tgt_phys.loc[tgt_split.test],
    )
    probe_r2 = {k: float(p.r2) for k, p in probes.items()}
    print("\n--- Target-region physics probes ---")
    for name, pr in probes.items():
        print(f"  {name}: R²={pr.r2:.3f}  RMSE={pr.rmse:.3f}  (n_test={pr.n_test})")

    # --- Source ↔ target alignment (optional) ---
    mmd_value: float | None = None
    silhouette_value: float | None = None
    if args.source_config is not None:
        src_cfg = load_run_config(args.source_config)
        src_loader = SteadLoader(src_cfg.loader)
        src_site = load_or_compute_site_features(src_loader)
        rng = np.random.default_rng(cfg.train.seed)
        all_src_names = src_loader.metadata["trace_name"].tolist()
        n_src = min(args.source_embed_sample, len(all_src_names))
        src_sample = [
            all_src_names[i] for i in rng.choice(len(all_src_names), size=n_src, replace=False)
        ]
        # Reuse the source-region cache built during pretraining. Keyed by
        # src_cfg.loader + full source trace set, so if pretraining ran on
        # the same config this is a cache hit; the src_sample is a subset.
        src_cache = WaveformCache(
            src_cfg.loader, trace_names=src_loader.metadata["trace_name"].tolist()
        )
        src_cache_available = src_cache.exists()
        if src_cache_available:
            print(f"[cache] source hit — {src_cache.data_path.name}", flush=True)
        else:
            print(
                "[cache] no source cache found; source embeddings will use "
                "HDF5+preprocess fallback (slow but one-shot)",
                flush=True,
            )
        src_ds = WaveformDataset(
            src_cfg.loader,
            src_loader.metadata,
            src_site,
            src_sample,
            waveform_cache=src_cache if src_cache_available else None,
        )
        src_loader_dl = build_loader(
            src_ds,
            cfg.train.batch_size,
            cfg.train.num_workers,
            prefetch_factor=cfg.train.prefetch_factor,
        )
        src_embed = extract_physics_embeddings(model, src_loader_dl, device)
        mmd_value = rbf_mmd2(src_embed, tgt_test_embed)
        combined = np.vstack([src_embed, tgt_test_embed])
        labels = np.array(
            ["source"] * len(src_embed) + ["target"] * len(tgt_test_embed)
        )
        silhouette_value = region_silhouette(combined, labels)
        print(
            f"\n--- Alignment ({cfg.loader.region} vs California) ---\n"
            f"  MMD²       = {mmd_value:.4f}   (lower = better overlap)\n"
            f"  silhouette = {silhouette_value:.4f}  (lower = regions mixed)"
        )

    # --- Build the transfer report and run few-shot + from-scratch stages ---
    report = TransferReport(
        target_region=cfg.loader.region,
        mmd_source_target=mmd_value,
        silhouette_source_target=silhouette_value,
        probe_r2=probe_r2,
    )

    from eqxfer.evaluation.transfer_eval import TransferStageResult

    report.stages.append(
        TransferStageResult(stage="zero_shot", n_adapt_events=0, panel=zs_panel)
    )
    append_result_row(
        _row_for_stage(cfg, exp_id, report.stages[-1], mmd_value, probe_r2)
    )

    # Events available for adaptation = training-split events (val/test
    # events excluded by construction). Pass the FULL target metadata to
    # the transfer functions so they can build val_ds and adapt_ds against
    # the same frame; the train_trace_names arg scopes event sampling.
    n_target_events = int(
        tgt_loader.metadata.loc[
            tgt_loader.metadata["trace_name"].isin(tgt_split.train),
            "source_id",
        ].nunique()
    )
    print(
        f"\nAdaptation pool: {len(tgt_split.train):,} traces from "
        f"{n_target_events} events"
    )

    for n in cfg.transfer.few_shot_n:
        if n > n_target_events:
            print(f"[skip] few-shot n={n} exceeds {n_target_events} available events")
            continue

        # Few-shot fine-tune.
        fs_result = few_shot_finetune(
            pretrained_state=pretrained_state,
            model_cfg=cfg.model_hparams,
            train_cfg=cfg.train,
            transfer_cfg=cfg.transfer,
            target_metadata=tgt_loader.metadata,
            target_site_features=tgt_site,
            loader_cfg=cfg.loader,
            n_events=n,
            train_trace_names=tgt_split.train,
            val_trace_names=tgt_split.val,
            test_loader=tgt_test_loader,
            device=device,
            exp_dir=exp_dir,
            waveform_cache=tgt_cache,
        )
        report.stages.append(fs_result)
        append_result_row(_row_for_stage(cfg, exp_id, fs_result, mmd_value, probe_r2))

        # From-scratch baseline at the same budget.
        fs0_result = from_scratch_baseline(
            model_cfg=cfg.model_hparams,
            train_cfg=cfg.train,
            transfer_cfg=cfg.transfer,
            target_metadata=tgt_loader.metadata,
            target_site_features=tgt_site,
            loader_cfg=cfg.loader,
            n_events=n,
            train_trace_names=tgt_split.train,
            val_trace_names=tgt_split.val,
            test_loader=tgt_test_loader,
            device=device,
            exp_dir=exp_dir,
            waveform_cache=tgt_cache,
        )
        report.stages.append(fs0_result)
        append_result_row(_row_for_stage(cfg, exp_id, fs0_result, None, {}))

    report.write(exp_dir / "transfer_report.json")
    (exp_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
    )
    print(f"\nTransfer report: {exp_dir / 'transfer_report.json'}")
    print("\n--- Summary ---")
    for stage in report.stages:
        p = stage.panel
        print(
            f"{stage.stage:30s}  n_adapt={stage.n_adapt_events:5d}  "
            f"MAE={p.mae:.4f}  acc±0.3={p.acc_0p3:.3f}  "
            f"MAE(M6+)={p.mae_m6plus:.4f} (n={p.n_m6plus})"
        )


if __name__ == "__main__":
    main()
