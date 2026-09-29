"""Precompute the MDN pseudo-label raster (``mdn_chla``) of every scene.

For every scene of the configured archives the MDN maps the seven 443-783 nm
bands to Chl-a; log10 of the clipped estimate is written as a float32
GeoTIFF beside the scene (NaN where any band is nonpositive) and registered
in the manifest row under ``products.mdn_chla``. Scenes that already have
one are skipped, so a rerun only fills gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from chla_prediction.config import PrecomputePseudoLabelsConfig
from chla_prediction.imagery.archive import REFLECTANCE_SCALE, read_bands
from chla_prediction.io import read_jsonl, write_jsonl
from chla_prediction.retrieval.chla import ChlaRetrieval, log10_chla
from chla_prediction.retrieval.mdn import MDN_BANDS

NAME = "mdn_chla"
SUFFIX = "_LOG10CHLA.tif"

__all__ = ["PrecomputeOutput", "precompute_pseudo_labels"]


@dataclass(frozen=True)
class PrecomputeOutput:
    n_written: int
    n_skipped: int


def _write_pseudo_label(scene_path: Path, log_chla: np.ndarray) -> Path:
    out_path = scene_path.with_name(scene_path.stem + SUFFIX)
    with rasterio.open(scene_path) as src:
        profile = src.profile
    profile.update(dtype="float32", count=1, nodata=np.nan, predictor=3)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(log_chla.astype(np.float32), 1)
        dst.set_band_description(1, "LOG10_CHLA")
    return out_path


def precompute_pseudo_labels(config_path: Path) -> PrecomputeOutput:
    config = PrecomputePseudoLabelsConfig.from_yaml(config_path)
    retrieve = ChlaRetrieval(config.mdn_weights, config.device)
    n_written = n_skipped = 0
    for entry in config.data.archive_entries():
        base = entry.manifest_path.parent
        rows = read_jsonl(entry.manifest_path)
        for row in rows:
            if NAME in row.get("products", {}):
                n_skipped += 1
                continue
            scene_path = base / row["local_path"]
            digital_numbers = read_bands(scene_path, MDN_BANDS)
            reflectance = digital_numbers.astype(np.float32) * REFLECTANCE_SCALE
            chla = retrieve(NAME, reflectance, MDN_BANDS, (digital_numbers > 0).all(axis=0))
            out_path = _write_pseudo_label(scene_path, log10_chla(chla, config.chla_range))
            row.setdefault("products", {})[NAME] = str(out_path.relative_to(base))
            n_written += 1
        write_jsonl(entry.manifest_path, rows)
        print(f"{entry.water_id}: {len(rows)} scenes", flush=True)
    return PrecomputeOutput(n_written=n_written, n_skipped=n_skipped)
