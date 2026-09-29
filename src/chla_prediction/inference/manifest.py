"""Prediction manifests: one JSONL row per forecast raster.

The manifest is the contract between every forecasting method and the
evaluation: a row names the sample, the prediction raster, its bands, and
the target scene it is scored against. Each method writes one manifest per
target water; a variant's manifests are merged before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from chla_prediction.imagery.archive import Scene
from chla_prediction.imagery.sequences import SequenceSample
from chla_prediction.io import read_jsonl, write_jsonl

__all__ = [
    "PredictionOutput",
    "merge_manifests",
    "prediction_row",
    "sample_id",
    "write_prediction_raster",
]


@dataclass(frozen=True)
class PredictionOutput:
    run_dir: Path
    manifest_path: Path
    prediction_count: int


def sample_id(sample: SequenceSample) -> str:
    return f"{sample.water_id}_{sample.origin_date:%Y%m%d}_{sample.target_date:%Y%m%d}"


def prediction_row(
    sample: SequenceSample, target: Scene, prediction_path: Path | None, bands: list[str], **fields: object
) -> dict:
    """One manifest row; ``fields`` (split, model name, checkpoint, ...) follow the sample's dates.

    Without a raster (``prediction_path=None``) the row describes a prediction scored in memory.
    """
    return {
        "sample_id": sample_id(sample),
        "water_id": sample.water_id,
        "origin_date": f"{sample.origin_date:%Y%m%d}",
        "target_date": f"{sample.target_date:%Y%m%d}",
        "horizon_days": sample.lead_days,
        **fields,
        "bands": bands,
        "prediction_path": None if prediction_path is None else str(prediction_path),
        "reference_path": str(target.path),
        "reference_scl_path": str(target.scl_path),
    }


def write_prediction_raster(
    path: Path, prediction: np.ndarray, target: Scene, bands: list[str], nodata: float | None = None
) -> None:
    """Write a ``[bands, H, W]`` float32 forecast on the grid of its target scene."""
    with rasterio.open(target.path) as src:
        profile = src.profile | {"count": prediction.shape[0], "dtype": "float32", "nodata": nodata}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prediction)
        for index, name in enumerate(bands, start=1):
            dst.set_band_description(index, name)


def merge_manifests(paths: list[Path], output: Path) -> int:
    """Concatenate one variant's per-water manifests; a missing one is an error.

    A variant is thus never scored on fewer waters than it was run on.
    """
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prediction manifests: {missing}")
    rows = [row for path in paths for row in read_jsonl(path)]
    write_jsonl(output, rows)
    return len(rows)
