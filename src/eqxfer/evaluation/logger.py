"""Append one row per run to experiments/results.csv. Also creates
experiments/<exp_id>/ for per-run artifacts."""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

RESULTS_CSV = Path("experiments/results.csv")
EXPERIMENTS_DIR = Path("experiments")


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def make_exp_id(rung: int) -> str:
    return f"rung{rung}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def make_exp_dir(exp_id: str) -> Path:
    path = EXPERIMENTS_DIR / exp_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_result_row(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} is missing — scaffold should have created it with headers"
        )
    with RESULTS_CSV.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    missing = [k for k in header if k not in row]
    for k in missing:
        row[k] = ""
    ordered = [row[col] for col in header]
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ordered)
