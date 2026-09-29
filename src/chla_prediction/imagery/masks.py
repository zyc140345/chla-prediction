"""Mask derivation from Sentinel-2 scenes and external water-body products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, label

# SCL classes treated as unusable observations: nodata, saturated/defective,
# cloud shadow, cloud medium/high probability, thin cirrus, snow/ice.
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)

__all__ = [
    "SCL_INVALID",
    "erode_shoreline",
    "water_extent_masks",
    "find_water_mask_band",
    "largest_water_body",
    "normalized_difference",
    "read_water_mask_band",
    "scl_valid_mask",
    "water_extent_mask",
]


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)``, NaN where the denominator vanishes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        index = (a - b) / (a + b)
    return np.where(np.isfinite(index), index, np.nan)


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels with a usable surface observation."""
    return ~np.isin(scl, SCL_INVALID)


def largest_water_body(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected water region, dropping neighboring water bodies.

    Each archive is a box around one water, so its largest component is that
    water. The center component is not used: on Hushan the center pixel lies
    outside the reservoir.
    """
    labels, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def erode_shoreline(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Drop the outermost ``pixels`` rings of a water mask.

    Shoreline pixels mix land adjacency and bottom reflectance. Diagonal
    neighbors count, so any pixel touching the boundary is removed.
    """
    if pixels <= 0:
        return mask
    return binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=pixels, border_value=0)


def water_extent_mask(raster: np.ndarray, threshold: float, shoreline_erosion_px: int) -> np.ndarray:
    """Threshold one band of an external water product, keep the water body, and erode the shoreline."""
    return erode_shoreline(largest_water_body(raster >= threshold), shoreline_erosion_px)


def water_extent_masks(archives, config) -> dict[str, np.ndarray]:
    """Fixed water-extent mask per water, on the water's own grid (tools/data/fetch_water_extent.py).

    ``s1_median_water`` is the share of training-period Sentinel-1 scenes
    with VV below -17.5 dB; unlike optical water indices it keeps eutrophic
    water.
    """
    directory = Path(config.water_mask_dir)
    masks = {}
    for archive in archives:
        raster = read_water_mask_band(directory, archive.water_id, config.water_mask_band)
        masks[archive.water_id] = water_extent_mask(raster, config.water_mask_threshold, config.shoreline_erosion_px)
    return masks


def find_water_mask_band(directory: Path, water_id: str, band: str) -> np.ndarray | None:
    """Read the named band from whichever candidate raster of this water carries it, or return None.

    Candidates come from different sources, one file each; a water may lack some.
    """
    for path in sorted(directory.glob(f"{water_id}_*.tif")):
        with rasterio.open(path) as src:
            if band in src.descriptions:
                return src.read(src.descriptions.index(band) + 1)
    return None


def read_water_mask_band(directory: Path, water_id: str, band: str) -> np.ndarray:
    """``find_water_mask_band`` for the evaluation, where a missing band is an error."""
    raster = find_water_mask_band(directory, water_id, band)
    if raster is None:
        raise FileNotFoundError(f"No water-extent raster in {directory} carries band {band!r} for {water_id}")
    return raster
