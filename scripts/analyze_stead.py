"""STEAD dataset analysis.

Reads STEAD metadata CSV(s) and reports geographic, magnitude, depth, SNR,
and trace-category breakdowns. Writes a JSON summary to experiments/.

Usage:
    python scripts/analyze_stead.py --stead-dir data/raw/stead
    python scripts/analyze_stead.py --csv data/raw/stead/merged.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_REGION_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "California": (32.5, 42.0, -124.5, -114.0),
    "Turkey":     (36.0, 42.5,   26.0,   45.0),
    "Greece":     (34.0, 42.0,   19.0,   30.0),
    "Japan":      (30.0, 46.0,  129.0,  146.0),
    "Chile":      (-56.0, -17.0, -76.0, -66.0),
}

MAG_BINS: list[tuple[float, float, str]] = [
    (2.5, 3.5, "M2.5-3.5"),
    (3.5, 4.5, "M3.5-4.5"),
    (4.5, 5.5, "M4.5-5.5"),
    (5.5, 6.5, "M5.5-6.5"),
    (6.5, 10.0, "M6.5+"),
]


@dataclass
class Summary:
    total_traces: int = 0
    earthquake_traces: int = 0
    noise_traces: int = 0
    unique_events: int = 0
    unique_stations: int = 0
    magnitude_types: dict[str, int] = field(default_factory=dict)
    magnitude_overall: dict[str, float] = field(default_factory=dict)
    magnitude_binned_overall: dict[str, int] = field(default_factory=dict)
    depth_summary_km: dict[str, float] = field(default_factory=dict)
    snr_summary_db: dict[str, float] = field(default_factory=dict)
    by_country: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_target_region: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the STEAD dataset.")
    p.add_argument("--stead-dir", type=Path, default=Path("data/raw/stead"),
                   help="Directory containing STEAD metadata (merged.csv or chunk*.csv).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Explicit path to a single STEAD metadata CSV. Overrides --stead-dir.")
    p.add_argument("--output", type=Path, default=None,
                   help="JSON summary output path. Default: experiments/stead_analysis_<ts>.json")
    p.add_argument("--top-countries", type=int, default=30,
                   help="How many countries to print in the country table.")
    p.add_argument("--no-reverse-geocode", action="store_true",
                   help="Skip country assignment (faster, still does target-region boxes).")
    return p.parse_args()


def find_csvs(stead_dir: Path, explicit_csv: Path | None) -> list[Path]:
    if explicit_csv is not None:
        if not explicit_csv.exists():
            sys.exit(f"CSV not found: {explicit_csv}")
        return [explicit_csv]

    if not stead_dir.exists():
        sys.exit(f"STEAD directory not found: {stead_dir}")

    candidates = sorted(stead_dir.glob("merged.csv"))
    if candidates:
        return candidates
    candidates = sorted(stead_dir.glob("chunk*.csv"))
    if candidates:
        return candidates
    candidates = sorted(stead_dir.glob("*.csv"))
    if candidates:
        return candidates

    sys.exit(f"No CSV files found in {stead_dir}")


def load_metadata(csv_paths: list[Path]) -> pd.DataFrame:
    print(f"Loading {len(csv_paths)} CSV file(s)...", flush=True)
    frames = []
    for p in csv_paths:
        print(f"  {p}", flush=True)
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.", flush=True)
    return df


def in_bbox(lat: pd.Series, lon: pd.Series, bbox: tuple[float, float, float, float]) -> pd.Series:
    lat_min, lat_max, lon_min, lon_max = bbox
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def assign_target_region(df: pd.DataFrame) -> pd.Series:
    region = pd.Series(index=df.index, dtype="object")
    lat = df["source_latitude"]
    lon = df["source_longitude"]
    for name, bbox in TARGET_REGION_BBOXES.items():
        mask = in_bbox(lat, lon, bbox)
        region = region.where(~mask, name)
    return region


def assign_countries(df: pd.DataFrame) -> pd.Series:
    try:
        import reverse_geocoder as rg
    except ImportError:
        print("  reverse_geocoder not installed; skipping per-country breakdown.", flush=True)
        print("  Install with: pip install reverse_geocoder", flush=True)
        return pd.Series(["<unknown>"] * len(df), index=df.index, dtype="object")

    print("Reverse-geocoding event locations (this takes a minute)...", flush=True)
    coords = list(zip(df["source_latitude"].fillna(0.0), df["source_longitude"].fillna(0.0)))
    results = rg.search(coords, mode=2)
    codes = pd.Series([r["cc"] for r in results], index=df.index)
    # Null out for rows with missing coords (noise traces)
    missing = df["source_latitude"].isna() | df["source_longitude"].isna()
    codes = codes.where(~missing, other="<none>")
    return codes


def magnitude_bin_label(m: float) -> str | None:
    if pd.isna(m):
        return None
    for lo, hi, label in MAG_BINS:
        if lo <= m < hi:
            return label
    return None


def summarize_magnitude(mags: pd.Series) -> dict[str, float]:
    mags = mags.dropna()
    if len(mags) == 0:
        return {}
    return {
        "count": int(len(mags)),
        "min": float(mags.min()),
        "p05": float(mags.quantile(0.05)),
        "median": float(mags.median()),
        "mean": float(mags.mean()),
        "p95": float(mags.quantile(0.95)),
        "max": float(mags.max()),
    }


def summarize_numeric(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def parse_snr(s: pd.Series) -> pd.Series:
    def parse_one(x: Any) -> float:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        # STEAD stores snr_db as a stringified list like "[11.2 9.5 14.1]"
        cleaned = str(x).strip("[]").replace(",", " ").split()
        try:
            vals = [float(v) for v in cleaned if v]
            return float(np.mean(vals)) if vals else np.nan
        except ValueError:
            return np.nan

    return s.apply(parse_one)


def build_subset_stats(sub: pd.DataFrame) -> dict[str, Any]:
    eq = sub[sub["trace_category"] == "earthquake_local"] if "trace_category" in sub.columns else sub
    binned: dict[str, int] = {label: 0 for _, _, label in MAG_BINS}
    labeled = eq["source_magnitude"].apply(magnitude_bin_label)
    for label, cnt in labeled.value_counts().items():
        if label in binned:
            binned[label] = int(cnt)

    return {
        "total_traces": int(len(sub)),
        "earthquake_traces": int(len(eq)),
        "unique_events": int(eq["source_id"].nunique()) if "source_id" in eq.columns else 0,
        "unique_stations": int(eq["receiver_code"].nunique()) if "receiver_code" in eq.columns else 0,
        "magnitude_summary": summarize_magnitude(eq["source_magnitude"]),
        "magnitude_binned": binned,
    }


def build_summary(df: pd.DataFrame, top_countries: int, do_rg: bool) -> Summary:
    s = Summary()
    s.total_traces = int(len(df))

    cat = df.get("trace_category")
    if cat is not None:
        s.earthquake_traces = int((cat == "earthquake_local").sum())
        s.noise_traces = int((cat == "noise").sum())
        eq_df = df[cat == "earthquake_local"]
    else:
        eq_df = df

    if "source_id" in eq_df.columns:
        s.unique_events = int(eq_df["source_id"].nunique())
    if "receiver_code" in eq_df.columns:
        s.unique_stations = int(eq_df["receiver_code"].nunique())

    if "source_magnitude_type" in eq_df.columns:
        s.magnitude_types = (
            eq_df["source_magnitude_type"].fillna("<unknown>").value_counts().astype(int).to_dict()
        )

    s.magnitude_overall = summarize_magnitude(eq_df["source_magnitude"])
    s.magnitude_binned_overall = {label: 0 for _, _, label in MAG_BINS}
    for label, cnt in eq_df["source_magnitude"].apply(magnitude_bin_label).value_counts().items():
        if label in s.magnitude_binned_overall:
            s.magnitude_binned_overall[label] = int(cnt)

    if "source_depth_km" in eq_df.columns:
        s.depth_summary_km = summarize_numeric(eq_df["source_depth_km"])

    if "snr_db" in eq_df.columns:
        s.snr_summary_db = summarize_numeric(parse_snr(eq_df["snr_db"]))

    region = assign_target_region(eq_df)
    for name in TARGET_REGION_BBOXES:
        sub = eq_df[region == name]
        s.by_target_region[name] = build_subset_stats(sub)

    if do_rg:
        countries = assign_countries(eq_df)
        counts = countries.value_counts()
        picked = counts.head(top_countries).index.tolist()
        for cc in picked:
            sub = eq_df[countries == cc]
            s.by_country[cc] = build_subset_stats(sub)

    return s


def print_report(s: Summary) -> None:
    print()
    print("=" * 72)
    print("STEAD DATASET ANALYSIS")
    print("=" * 72)

    print(f"\nTotal traces:       {s.total_traces:>12,}")
    print(f"  earthquake_local: {s.earthquake_traces:>12,}")
    print(f"  noise:            {s.noise_traces:>12,}")
    print(f"Unique events:      {s.unique_events:>12,}")
    print(f"Unique stations:    {s.unique_stations:>12,}")

    if s.magnitude_overall:
        print("\nMagnitude (earthquake traces):")
        for k, v in s.magnitude_overall.items():
            print(f"  {k:>6}: {v}" if isinstance(v, int) else f"  {k:>6}: {v:.2f}")

    if s.magnitude_binned_overall:
        print("\nMagnitude bins (trace counts, earthquake traces):")
        total = sum(s.magnitude_binned_overall.values()) or 1
        for label, cnt in s.magnitude_binned_overall.items():
            pct = 100.0 * cnt / total
            print(f"  {label:>10}: {cnt:>10,}  ({pct:5.2f}%)")

    if s.magnitude_types:
        print("\nMagnitude types:")
        for mt, cnt in sorted(s.magnitude_types.items(), key=lambda x: -x[1]):
            print(f"  {mt:>10}: {cnt:>10,}")

    if s.depth_summary_km:
        print("\nDepth (km, earthquake traces):")
        for k, v in s.depth_summary_km.items():
            print(f"  {k:>6}: {v}" if isinstance(v, int) else f"  {k:>6}: {v:.2f}")

    if s.snr_summary_db:
        print("\nSNR (dB, mean across components):")
        for k, v in s.snr_summary_db.items():
            print(f"  {k:>6}: {v}" if isinstance(v, int) else f"  {k:>6}: {v:.2f}")

    print("\n" + "-" * 72)
    print("TARGET REGIONS (bounding-box assignment; paper focus)")
    print("-" * 72)
    for name, stats in s.by_target_region.items():
        print(f"\n[{name}]")
        print(f"  traces:   {stats['earthquake_traces']:>10,}")
        print(f"  events:   {stats['unique_events']:>10,}")
        print(f"  stations: {stats['unique_stations']:>10,}")
        if stats["magnitude_summary"]:
            ms = stats["magnitude_summary"]
            print(f"  mag range: {ms['min']:.2f} – {ms['max']:.2f}  (median {ms['median']:.2f})")
        print(f"  mag bins (traces):")
        for label, cnt in stats["magnitude_binned"].items():
            print(f"    {label:>10}: {cnt:>10,}")

    if s.by_country:
        print("\n" + "-" * 72)
        print(f"TOP COUNTRIES (by trace count, earthquake_local only)")
        print("-" * 72)
        print(f"{'CC':>4}  {'traces':>12}  {'events':>10}  {'stations':>10}  "
              f"{'mag_min':>8}  {'mag_max':>8}")
        rows = sorted(s.by_country.items(), key=lambda kv: -kv[1]["earthquake_traces"])
        for cc, stats in rows:
            ms = stats.get("magnitude_summary") or {}
            mn = f"{ms.get('min', float('nan')):.2f}" if ms else "   -  "
            mx = f"{ms.get('max', float('nan')):.2f}" if ms else "   -  "
            print(f"{cc:>4}  {stats['earthquake_traces']:>12,}  "
                  f"{stats['unique_events']:>10,}  {stats['unique_stations']:>10,}  "
                  f"{mn:>8}  {mx:>8}")

    print()


def save_summary(s: Summary, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(s.to_dict(), f, indent=2, default=str)
    print(f"Wrote JSON summary: {output}")


def main() -> None:
    args = parse_args()
    csvs = find_csvs(args.stead_dir, args.csv)
    df = load_metadata(csvs)

    required = ["source_latitude", "source_longitude", "source_magnitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Missing required STEAD columns: {missing}. Got: {list(df.columns)}")

    s = build_summary(df, top_countries=args.top_countries, do_rg=not args.no_reverse_geocode)
    print_report(s)

    output = args.output
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("experiments") / f"stead_analysis_{ts}.json"
    save_summary(s, output)


if __name__ == "__main__":
    main()
