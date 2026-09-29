r"""Export the Chl-a field maps of the field-comparison figure.

For each sample, retrieves the MDN pseudo-label map of the real target scene
and turns each model's stored full-scene prediction into a Chl-a map on the
same grid: pseudo-label forecasts directly, reflectance forecasts through the
MDN. Writes one compressed ``.npz`` per sample with the maps, the mask of
valid pixels (water extent and observed) and the RMSE_log of each map, for
``plot_field_comparison.py`` to render. It reads the archive rasters and the
predictions, so run it where those are.

Usage:
    python tools/figures/export_field_maps.py --config configs/paper/evaluate.yaml \
        --models persistence,climatology,rf_pixel,xgboost_pixel,scratch_convlstm,pix2pix_zero,ours_adapted \
        --samples baogu_20250607_20250915,loweswater_20250508_20250513,georges_20250920_20251002,wentzel_20241119_20241124,hushan_20250106_20250111 \
        --output-dir outputs/runs/field_maps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from chla_prediction.config import EvaluateForecastConfig, pseudo_label_of
from chla_prediction.evaluation.scoring import rmse_log
from chla_prediction.imagery.archive import read_reflectance
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.waters import load_waters
from chla_prediction.io import read_jsonl
from chla_prediction.retrieval.chla import RETRIEVAL_BANDS, ChlaRetrieval


def model_chla_map(row: dict, valid: np.ndarray, retrieve: ChlaRetrieval) -> np.ndarray:
    """Chl-a (mg/m^3) map of one stored prediction, NaN outside ``valid``."""
    with rasterio.open(row["prediction_path"]) as src:
        prediction = src.read().astype(np.float32)
    bands = list(row["bands"])
    if pseudo_label_of(bands) is not None:
        chla = np.where(valid, 10.0 ** prediction[0], np.nan)
    else:
        chla = retrieve("mdn_chla", prediction, bands, valid)
    return chla.astype(np.float32)


def map_rmse_log(chla: np.ndarray, reference: np.ndarray, chla_range: tuple[float, float]) -> float:
    """RMSE_log over the pixels finite in both maps."""
    usable = np.isfinite(chla) & np.isfinite(reference)
    return rmse_log(chla[usable], reference[usable], chla_range)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Evaluation YAML the maps should agree with")
    parser.add_argument("--models", required=True, help="Comma-separated model names from prediction_manifests")
    parser.add_argument("--samples", required=True, help="Comma-separated sample ids to export")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = EvaluateForecastConfig.from_yaml(args.config)
    models = args.models.split(",")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrieve = ChlaRetrieval(config.mdn_weights, "cuda" if torch.cuda.is_available() else "cpu")
    archives, _ = load_waters(config.data)
    water_masks = water_extent_masks(archives, config)
    rows = {name: {row["sample_id"]: row for row in read_jsonl(config.prediction_manifests[name])} for name in models}

    bands = RETRIEVAL_BANDS["mdn_chla"]
    for sample_id in args.samples.split(","):
        base = rows[models[0]][sample_id]
        reflectance, observed = read_reflectance(base["reference_path"], base["reference_scl_path"], bands)
        valid = water_masks[base["water_id"]] & observed
        reference = retrieve("mdn_chla", reflectance, bands, valid)
        maps = {name: model_chla_map(rows[name][sample_id], valid, retrieve) for name in models}
        meta = {
            "sample_id": sample_id,
            "water_id": base["water_id"],
            "origin_date": base["origin_date"],
            "target_date": base["target_date"],
            "horizon_days": base["horizon_days"],
            "models": models,
            "chla_range": list(config.chla_range),
            "rmse_log": {name: map_rmse_log(chla, reference, config.chla_range) for name, chla in maps.items()},
        }
        output = args.output_dir / f"{sample_id}.npz"
        np.savez_compressed(
            output,
            reference=reference.astype(np.float32),
            valid=valid,
            meta=np.array(json.dumps(meta)),
            **{f"model_{name}": chla for name, chla in maps.items()},
        )
        print(f"wrote {output} ({int(valid.sum())} valid pixels)")


if __name__ == "__main__":
    main()
