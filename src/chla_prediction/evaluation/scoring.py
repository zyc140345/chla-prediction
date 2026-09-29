"""Score forecasts against their target scenes inside the water-extent masks.

Each sample gets output-space metrics (``rmse``, ``mae``, ``bias``, plus
``ndci_mae`` for reflectance forecasts) and Chl-a metrics (``rmse_log`` and
``chla_*``). For the Chl-a metrics, predicted and observed reflectance both
pass through the ``chla_axis`` retrieval; pseudo-label forecasts enter them
directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from chla_prediction.config import EvaluateForecastConfig, pseudo_label_of
from chla_prediction.imagery.archive import SceneArchive, read_reflectance
from chla_prediction.imagery.sequences import SequenceSample
from chla_prediction.inference.manifest import prediction_row
from chla_prediction.recipe import CHLA_RANGE
from chla_prediction.retrieval.chla import RETRIEVAL_BANDS, ChlaRetrieval, log10_chla
from chla_prediction.retrieval.empirical import ndci, three_band_chla

OUTPUT_METRICS = ["rmse", "mae", "bias", "ndci_mae"]
CHLA_METRICS = ["chla_rmse", "chla_mae", "chla_pearson_r", "rmse_log"]

__all__ = [
    "CHLA_METRICS",
    "OUTPUT_METRICS",
    "SampleScorer",
    "chla_metrics",
    "rmse_log",
    "score_split",
    "write_scores",
]


def rmse_log(pred_chla: np.ndarray, true_chla: np.ndarray, chla_range: tuple[float, float] = CHLA_RANGE) -> float:
    """Root-mean-square error of log10 Chl-a, both clipped to ``chla_range``."""
    return float(np.sqrt(np.mean((log10_chla(pred_chla, chla_range) - log10_chla(true_chla, chla_range)) ** 2)))


def chla_metrics(pred_chla: np.ndarray, true_chla: np.ndarray, chla_range: tuple[float, float]) -> dict:
    """Errors and correlation between Chl-a vectors clipped to ``chla_range``."""
    pred = np.clip(pred_chla, *chla_range)
    true = np.clip(true_chla, *chla_range)
    difference = pred - true
    pearson_r = float("nan")
    if pred.size >= 2 and np.std(pred) > 0 and np.std(true) > 0:
        pearson_r = float(np.corrcoef(pred, true)[0, 1])
    return {
        "chla_rmse": float(np.sqrt(np.mean(difference**2))),
        "chla_mae": float(np.mean(np.abs(difference))),
        "chla_pearson_r": pearson_r,
        "rmse_log": rmse_log(pred, true, chla_range),
    }


class SampleScorer:
    """Score one prediction-manifest row against its target scene inside a water-extent mask."""

    def __init__(self, config: EvaluateForecastConfig):
        self.chla_range = config.chla_range
        self.axis = f"{config.chla_axis}_chla"
        self.retrieve = ChlaRetrieval(config.mdn_weights, "cuda" if torch.cuda.is_available() else "cpu")

    def score(self, row: dict, water_mask: np.ndarray) -> dict | None:
        """Score the prediction raster a manifest row names."""
        with rasterio.open(row["prediction_path"]) as src:
            prediction = src.read().astype(np.float32)
        return self.score_array(prediction, row, water_mask)

    def score_array(self, prediction: np.ndarray, row: dict, water_mask: np.ndarray) -> dict | None:
        """Score a ``[bands, H, W]`` float32 prediction; None when no mask pixel is usable."""
        bands = list(row["bands"])
        pseudo_label = pseudo_label_of(bands)
        reference_bands = RETRIEVAL_BANDS[pseudo_label] if pseudo_label else bands
        reference_bands = list(dict.fromkeys(reference_bands + RETRIEVAL_BANDS[self.axis]))
        reference, observed = read_reflectance(row["reference_path"], row["reference_scl_path"], reference_bands)

        valid = water_mask & observed & np.isfinite(prediction).all(axis=0)
        if "three_band_chla" in (pseudo_label, self.axis):
            valid &= np.isfinite(three_band_chla(reference, reference_bands))
        if not valid.any():
            return None
        # Coverage is relative to the water-extent mask, not the frame: the
        # water fills from about 5% (Hushan) to a third of its box.
        metrics = {
            "sample_id": row["sample_id"],
            "water_id": row["water_id"],
            "origin_date": row["origin_date"],
            "target_date": row["target_date"],
            "horizon_days": row["horizon_days"],
            "valid_fraction": round(float(valid.mean()), 4),
            "water_mask_fraction": round(float(valid.sum() / water_mask.sum()), 4),
            "n_valid_pixels": int(valid.sum()),
            "n_water_mask_pixels": int(water_mask.sum()),
        }
        if pseudo_label is not None:
            # Target: the forecast pseudo-label of the target scene, whatever the Chl-a axis.
            target = np.full(prediction.shape, np.nan, dtype=np.float32)
            own = self.retrieve(pseudo_label, reference, reference_bands, valid)[valid]
            target[0][valid] = log10_chla(own, self.chla_range)
            metrics["ndci_mae"] = float("nan")
        else:
            target = reference[[reference_bands.index(band) for band in bands]]
            metrics["ndci_mae"] = float(np.mean(np.abs(ndci(prediction, bands)[valid] - ndci(target, bands)[valid])))
        difference = (prediction - target)[:, valid]
        metrics.update(
            rmse=float(np.sqrt(np.mean(difference**2))),
            mae=float(np.mean(np.abs(difference))),
            bias=float(np.mean(difference)),
        )

        true_chla = self.retrieve(self.axis, reference, reference_bands, valid)[valid]
        if pseudo_label is not None:
            pred_chla = 10.0 ** prediction[0][valid]
        else:
            axis_bands = [bands.index(band) for band in RETRIEVAL_BANDS[self.axis]]
            positive = valid & (prediction[axis_bands] > 0).all(axis=0)
            pred_chla = self.retrieve(self.axis, prediction, bands, positive)[valid]
        usable = np.isfinite(pred_chla) & np.isfinite(true_chla)
        metrics.update(chla_metrics(pred_chla[usable], true_chla[usable], self.chla_range))
        metrics["chla_valid_fraction"] = round(float(usable.mean()), 4)
        return metrics


def score_split(
    predict: Callable[[SceneArchive, SequenceSample], np.ndarray],
    archives: list[SceneArchive],
    splits: dict[str, dict[str, list[SequenceSample]]],
    split: str,
    bands: list[str],
    scorer: SampleScorer,
    water_masks: dict[str, np.ndarray],
    min_valid_fraction: float,
) -> tuple[pd.DataFrame, int]:
    """Forecast and score every sample of one split in memory.

    Returns the scores of the samples with enough coverage and the number of candidate samples.
    """
    rows, candidates = [], 0
    for archive in archives:
        for sample in splits[archive.water_id][split]:
            candidates += 1
            row = prediction_row(sample, archive.scenes[sample.target_index], None, bands)
            metrics = scorer.score_array(predict(archive, sample), row, water_masks[archive.water_id])
            if metrics is not None and metrics["water_mask_fraction"] >= min_valid_fraction:
                rows.append(metrics)
    return pd.DataFrame(rows), candidates


def write_scores(frame: pd.DataFrame, output: Path, **meta: object) -> None:
    """Write per-sample scores to ``output`` (CSV) and ``meta`` to the JSON file beside it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{output}: {meta['scored_sequences']}/{meta['candidate_sequences']} scenes", flush=True)
