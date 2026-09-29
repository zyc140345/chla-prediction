"""Scene archives built from STAC download manifests."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio

from chla_prediction.imagery.masks import scl_valid_mask
from chla_prediction.io import read_jsonl

REFLECTANCE_SCALE = 1.0e-4

__all__ = [
    "REFLECTANCE_SCALE",
    "Scene",
    "SceneArchive",
    "load_pseudo_label",
    "load_reflectance",
    "observed_pixels",
    "read_bands",
    "read_reflectance",
    "read_scl",
    "scene_shape",
]


@dataclass(frozen=True)
class Scene:
    water_id: str
    scene_date: date
    path: Path
    scl_path: Path
    cloud_fraction: float
    crs: str
    # Precomputed pseudo-label rasters by name (retrieval/precompute.py), e.g. ``mdn_chla``.
    pseudo_labels: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneArchive:
    water_id: str
    scenes: tuple[Scene, ...]
    # Source manifest; per-water rasters such as the climatology live next to it.
    manifest_path: Path | None = None

    @classmethod
    def from_stac_manifest(cls, manifest_path: Path, water_id: str) -> SceneArchive:
        """Read a STAC download manifest, whose paths are relative to its directory."""
        base = Path(manifest_path).parent
        scenes = [
            Scene(
                water_id=water_id,
                scene_date=datetime.strptime(row["scene_date"], "%Y%m%d").date(),
                path=base / row["local_path"],
                scl_path=base / row["scl_path"],
                cloud_fraction=float(row["roi_cloud_shadow_fraction"]),
                crs=str(row["crs"]),
                pseudo_labels={name: base / raw for name, raw in row.get("products", {}).items()},
            )
            for row in read_jsonl(manifest_path)
        ]
        if not scenes:
            raise ValueError(f"Archive manifest has no scenes: {manifest_path}")
        scenes.sort(key=lambda scene: scene.scene_date)
        return cls(water_id=water_id, scenes=tuple(scenes), manifest_path=Path(manifest_path))

    def with_scenes(self, scenes: tuple[Scene, ...]) -> SceneArchive:
        return SceneArchive(water_id=self.water_id, scenes=scenes, manifest_path=self.manifest_path)

    def filtered(self, max_cloud_fraction: float) -> SceneArchive:
        """Keep the scenes with at most this cloud fraction; the result may be empty."""
        return self.with_scenes(tuple(scene for scene in self.scenes if scene.cloud_fraction <= max_cloud_fraction))

    def majority_crs(self) -> SceneArchive:
        """Keep the scenes on the most common CRS, so every frame shares one grid.

        Waters straddling a UTM zone boundary have scenes on two grids.
        """
        crs_counts = Counter(scene.crs for scene in self.scenes)
        majority = crs_counts.most_common(1)[0][0]
        if len(crs_counts) > 1:
            dropped = sum(count for crs, count in crs_counts.items() if crs != majority)
            warnings.warn(f"{self.water_id}: dropping {dropped} scenes not in majority CRS {majority}", stacklevel=2)
        return self.with_scenes(tuple(scene for scene in self.scenes if scene.crs == majority))


def _rio_window(window: tuple[slice, slice] | None) -> rasterio.windows.Window | None:
    return rasterio.windows.Window.from_slices(*window) if window is not None else None


def read_bands(path: str | Path, bands: list[str], window: tuple[slice, slice] | None = None) -> np.ndarray:
    """Read the named bands, located by the raster's band descriptions.

    ``window`` is a ``(rows, cols)`` slice pair; only that region is read.
    """
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        return src.read([names.index(band) + 1 for band in bands], window=_rio_window(window))


def read_scl(path: str | Path, window: tuple[slice, slice] | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, window=_rio_window(window))


def observed_pixels(digital_numbers: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """Pixels with nonzero DNs in every band and a usable SCL class."""
    return (digital_numbers > 0).all(axis=0) & scl_valid_mask(scl)


def read_reflectance(
    path: str | Path, scl_path: str | Path, bands: list[str], window: tuple[slice, slice] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read bands as float32 reflectance with the mask of observed pixels."""
    digital_numbers = read_bands(path, bands, window)
    reflectance = digital_numbers.astype(np.float32) * REFLECTANCE_SCALE
    return reflectance, observed_pixels(digital_numbers, read_scl(scl_path, window))


def load_reflectance(
    scene: Scene, bands: list[str], window: tuple[slice, slice] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    return read_reflectance(scene.path, scene.scl_path, bands, window)


def load_pseudo_label(scene: Scene, name: str, window: tuple[slice, slice] | None = None) -> np.ndarray:
    """``[1, H, W]`` float32 pseudo-label raster, NaN where it is undefined."""
    with rasterio.open(scene.pseudo_labels[name]) as src:
        return src.read(window=_rio_window(window)).astype(np.float32)


def scene_shape(scene: Scene) -> tuple[int, int]:
    """Raster ``(height, width)`` from the file header."""
    with rasterio.open(scene.path) as src:
        return src.height, src.width
