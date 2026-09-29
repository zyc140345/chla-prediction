"""Per-pixel seasonal means of a pseudo-label in 46 January-anchored eight-day bins.

Following Wu et al. (2026), doi:10.3390/rs18121904: the observed pixels are
averaged within each year and bin, then the available years are averaged
with equal weight. Only training-period scenes contribute, and empty bins
stay NaN. The last bin holds the final five days (six in a leap year).
The Climatology reference forecast reads the bin of each target date.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from chla_prediction.config import DataConfig
from chla_prediction.imagery.archive import SceneArchive, load_pseudo_label, read_scl
from chla_prediction.imagery.masks import scl_valid_mask

__all__ = [
    "SEASONAL_BINS",
    "Climatology",
    "climatology_path",
    "compute_climatology",
    "load_climatology",
    "seasonal_bin_index",
    "water_climatology",
    "write_climatology",
]

SEASONAL_BINS = 46


def seasonal_bin_index(day: date) -> int:
    return min((day.timetuple().tm_yday - 1) // 8, SEASONAL_BINS - 1)


def climatology_path(archive: SceneArchive, product: str) -> Path:
    return archive.manifest_path.parent / f"climatology_{product}_8day_mean.tif"


@dataclass(frozen=True)
class Climatology:
    full: np.ndarray  # [SEASONAL_BINS, H, W]

    def base(self, day: date) -> np.ndarray:
        """``[H, W]`` climatology in the bin of ``day``."""
        return self.full[seasonal_bin_index(day)]


def _nanmean(frames: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not frames:
        return np.full(shape, np.nan, dtype=np.float32)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # Pixels never valid in this bin stay NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(frames), axis=0)


def compute_climatology(archive: SceneArchive, product: str, train_end: date) -> Climatology:
    scenes = [s for s in archive.scenes if s.scene_date <= train_end and product in s.pseudo_labels]
    if not scenes:
        raise ValueError(f"{archive.water_id}: no training-period scenes with product {product}")
    by_bin: dict[int, list] = {}
    for scene in scenes:
        by_bin.setdefault(seasonal_bin_index(scene.scene_date), []).append(scene)
    years = sorted({scene.scene_date.year for scene in scenes})
    shape = load_pseudo_label(scenes[0], product)[0].shape
    full = np.full((SEASONAL_BINS, *shape), np.nan, dtype=np.float32)
    for bin_index, members in by_bin.items():
        frames = []
        for scene in members:
            frame = load_pseudo_label(scene, product)[0]
            frames.append(np.where(scl_valid_mask(read_scl(scene.scl_path)), frame, np.nan))
        annual = {
            year: _nanmean([f for f, s in zip(frames, members, strict=True) if s.scene_date.year == year], shape)
            for year in years
        }
        full[bin_index] = _nanmean(list(annual.values()), shape)
    return Climatology(full)


def write_climatology(archive: SceneArchive, product: str, train_end: date, out_path: Path) -> Path:
    climatology = compute_climatology(archive, product, train_end)
    with rasterio.open(archive.scenes[0].path) as src:
        profile = src.profile
    profile.update(dtype="float32", count=SEASONAL_BINS, nodata=np.nan, predictor=3)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(climatology.full)
        dst.update_tags(aggregation="annual_equal_mean", period_days=8, train_end=str(train_end))
        for i in range(SEASONAL_BINS):
            dst.set_band_description(i + 1, f"period_{i:02d}")
    return out_path


def load_climatology(path: Path) -> Climatology:
    with rasterio.open(path) as src:
        return Climatology(src.read(list(range(1, SEASONAL_BINS + 1))).astype(np.float32))


def water_climatology(archive: SceneArchive, data: DataConfig) -> Climatology | None:
    """The water's climatology of the forecast pseudo-label, if ``data.climatology_base`` asks for it."""
    return load_climatology(climatology_path(archive, data.forecast_product)) if data.climatology_base else None
