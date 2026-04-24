"""Site-local geological features fed only to the RegionalSiteEncoder.

Per CLAUDE.md hard rule #5, nothing in this module ever reaches the
UniversalPhysicsEncoder. The encoder's contract (see models/split_transfer.py)
accepts the waveform tensor only.

Features chosen so they're physically meaningful in any tectonic regime
— no lat/lon, no station-ID one-hots, no regional labels. A Chilean
station with Vs30 = 350 m/s should look like a Greek station with the
same Vs30 to the site encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from ..features.pd_tauc import InstrumentKind, infer_instrument_kind


# ---------------------------------------------------------------------------
# Vs30 (USGS global Vs30, ~30 arcsec grid, HDF5-backed NetCDF4).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vs30Grid:
    """Nearest-neighbor sampler over a regular lat/lon grid of Vs30 values.

    Fully loaded into memory only for the bbox we need — the global grid is
    ~600 MB and loading it all into RAM would be wasteful."""

    lat: np.ndarray  # (H,) ascending
    lon: np.ndarray  # (W,) ascending
    z: np.ndarray    # (H, W) float32 m/s, NaN = no data

    @classmethod
    def load(
        cls,
        path: Path | str,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> "Vs30Grid":
        """Load the Vs30 grid. If bbox=(lat_min, lat_max, lon_min, lon_max) is
        given, only that window is read into memory."""
        with h5py.File(path, "r") as f:
            lat = np.asarray(f["lat"][:], dtype=np.float64)
            lon = np.asarray(f["lon"][:], dtype=np.float64)
            if bbox is None:
                z = np.asarray(f["z"][:], dtype=np.float32)
                return cls(lat=lat, lon=lon, z=z)
            lat_min, lat_max, lon_min, lon_max = bbox
            i0, i1 = np.searchsorted(lat, [lat_min, lat_max])
            j0, j1 = np.searchsorted(lon, [lon_min, lon_max])
            i0, i1 = max(0, int(i0) - 1), min(len(lat), int(i1) + 1)
            j0, j1 = max(0, int(j0) - 1), min(len(lon), int(j1) + 1)
            z = np.asarray(f["z"][i0:i1, j0:j1], dtype=np.float32)
            return cls(lat=lat[i0:i1], lon=lon[j0:j1], z=z)

    def sample(self, lat: float, lon: float) -> float:
        i = int(np.clip(np.searchsorted(self.lat, lat), 1, len(self.lat) - 1))
        j = int(np.clip(np.searchsorted(self.lon, lon), 1, len(self.lon) - 1))
        if abs(self.lat[i - 1] - lat) < abs(self.lat[i] - lat):
            i = i - 1
        if abs(self.lon[j - 1] - lon) < abs(self.lon[j] - lon):
            j = j - 1
        v = float(self.z[i, j])
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"Vs30 lookup at (lat={lat:.3f}, lon={lon:.3f}) returned "
                f"non-positive/non-finite value {v}"
            )
        return v


# ---------------------------------------------------------------------------
# CRUST1.0 (1°x1° global crustal model, classic NetCDF).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Crust1Grid:
    lat: np.ndarray
    lon: np.ndarray
    upper_crust_vp: np.ndarray      # (lat, lon) km/s
    sediment_thickness: np.ndarray  # (lat, lon) km (total upper+mid+lower sediments)

    @classmethod
    def load(cls, path: Path | str) -> "Crust1Grid":
        with netcdf_file(str(path), "r", mmap=False) as f:
            lat = np.asarray(f.variables["latitude"][:], dtype=np.float64).copy()
            lon = np.asarray(f.variables["longitude"][:], dtype=np.float64).copy()
            ucv = np.asarray(f.variables["upper_crust_vp"][:], dtype=np.float64).copy()
            ust = np.asarray(
                f.variables["upper_sediments_thickness"][:], dtype=np.float64
            ).copy()
            mst = np.asarray(
                f.variables["middle_sediments_thickness"][:], dtype=np.float64
            ).copy()
            lst = np.asarray(
                f.variables["lower_sediments_thickness"][:], dtype=np.float64
            ).copy()
        sed_thick = ust + mst + lst
        order_lat = np.argsort(lat)
        order_lon = np.argsort(lon)
        return cls(
            lat=lat[order_lat],
            lon=lon[order_lon],
            upper_crust_vp=ucv[np.ix_(order_lat, order_lon)],
            sediment_thickness=sed_thick[np.ix_(order_lat, order_lon)],
        )

    def _idx(self, lat: float, lon: float) -> tuple[int, int]:
        i = int(np.clip(np.searchsorted(self.lat, lat), 1, len(self.lat) - 1))
        j = int(np.clip(np.searchsorted(self.lon, lon), 1, len(self.lon) - 1))
        if abs(self.lat[i - 1] - lat) < abs(self.lat[i] - lat):
            i -= 1
        if abs(self.lon[j - 1] - lon) < abs(self.lon[j] - lon):
            j -= 1
        return i, j

    def sample_vp(self, lat: float, lon: float) -> float:
        i, j = self._idx(lat, lon)
        v = float(self.upper_crust_vp[i, j])
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"CRUST1 upper-crust Vp at (lat={lat:.2f}, lon={lon:.2f}) "
                f"is non-positive/non-finite: {v}"
            )
        return v

    def sample_sediment_thickness(self, lat: float, lon: float) -> float:
        i, j = self._idx(lat, lon)
        v = float(self.sediment_thickness[i, j])
        if not np.isfinite(v):
            raise ValueError(
                f"CRUST1 sediment thickness at (lat={lat:.2f}, lon={lon:.2f}) "
                f"is non-finite"
            )
        return max(v, 0.0)


# ---------------------------------------------------------------------------
# NEHRP site class from Vs30 (BSSC 2003 / ASCE 7-16 thresholds).
# ---------------------------------------------------------------------------

NEHRP_CLASSES: tuple[str, ...] = ("A", "B", "C", "D", "E")


def nehrp_class(vs30: float) -> str:
    if vs30 > 1500.0:
        return "A"
    if vs30 > 760.0:
        return "B"
    if vs30 > 360.0:
        return "C"
    if vs30 > 180.0:
        return "D"
    return "E"


def nehrp_onehot(vs30: float) -> np.ndarray:
    out = np.zeros(len(NEHRP_CLASSES), dtype=np.float32)
    out[NEHRP_CLASSES.index(nehrp_class(vs30))] = 1.0
    return out


def instrument_onehot(kind: InstrumentKind) -> np.ndarray:
    if kind == "velocity":
        return np.array([1.0, 0.0], dtype=np.float32)
    if kind == "acceleration":
        return np.array([0.0, 1.0], dtype=np.float32)
    raise ValueError(f"unknown instrument kind: {kind!r}")


# ---------------------------------------------------------------------------
# Per-trace site feature vector.
# ---------------------------------------------------------------------------

SITE_FEATURE_NAMES: tuple[str, ...] = (
    "log10_vs30",
    "upper_crust_vp_kms",
    "sediment_thickness_km",
    "nehrp_A",
    "nehrp_B",
    "nehrp_C",
    "nehrp_D",
    "nehrp_E",
    "inst_velocity",
    "inst_acceleration",
)
SITE_FEATURE_DIM = len(SITE_FEATURE_NAMES)


def site_feature_vector(
    receiver_latitude: float,
    receiver_longitude: float,
    receiver_type: str,
    vs30_grid: Vs30Grid,
    crust1_grid: Crust1Grid,
) -> np.ndarray:
    vs30 = vs30_grid.sample(receiver_latitude, receiver_longitude)
    vp = crust1_grid.sample_vp(receiver_latitude, receiver_longitude)
    sed = crust1_grid.sample_sediment_thickness(receiver_latitude, receiver_longitude)
    kind = infer_instrument_kind(receiver_type)
    out = np.concatenate(
        [
            np.array([np.log10(vs30)], dtype=np.float32),
            np.array([vp], dtype=np.float32),
            np.array([sed], dtype=np.float32),
            nehrp_onehot(vs30),
            instrument_onehot(kind),
        ]
    )
    if out.shape != (SITE_FEATURE_DIM,):
        raise AssertionError(f"site feature shape mismatch: {out.shape}")
    return out


def compute_site_features_table(
    metadata: pd.DataFrame,
    vs30_path: Path | str,
    crust1_path: Path | str,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    """Return a DataFrame indexed by trace_name with one column per feature
    in SITE_FEATURE_NAMES. Fails loudly if any lookup errors."""
    vs30_grid = Vs30Grid.load(vs30_path, bbox=bbox)
    crust1_grid = Crust1Grid.load(crust1_path)

    rows: list[dict[str, float]] = []
    index: list[str] = []
    for _, r in metadata.iterrows():
        vec = site_feature_vector(
            float(r["receiver_latitude"]),
            float(r["receiver_longitude"]),
            str(r["receiver_type"]),
            vs30_grid,
            crust1_grid,
        )
        rows.append({name: float(vec[i]) for i, name in enumerate(SITE_FEATURE_NAMES)})
        index.append(str(r["trace_name"]))
    return pd.DataFrame(rows, index=pd.Index(index, name="trace_name"))
