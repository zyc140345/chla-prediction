"""Per-pixel random forest and XGBoost baselines on lagged pseudo-labels.

Each pixel of a sample is one row: the log10 Chl-a pseudo-label in every
input slot, the days from each slot to the target, the forecast lead, and
the target date's season. One model is fitted per water on its adaptation
set and applied to every pixel of its evaluation scenes; the forecasts are
written to a prediction manifest like the networks'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from chla_prediction.config import PRODUCT_CHANNELS, DataConfig, TabularBaselineConfig
from chla_prediction.imagery.archive import SceneArchive
from chla_prediction.imagery.dataset import load_input_frames, load_model_frames
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.sequences import SequenceSample
from chla_prediction.imagery.waters import load_waters
from chla_prediction.inference.manifest import PredictionOutput, prediction_row, sample_id, write_prediction_raster
from chla_prediction.io import write_jsonl

__all__ = ["fit_water", "predict_sample", "run_tabular_baseline"]

Regressor = XGBRegressor | RandomForestRegressor


def _sample_features(
    archive: SceneArchive, sample: SequenceSample, data: DataConfig
) -> tuple[np.ndarray, tuple[int, int]]:
    """Feature matrix ``[pixels, features]`` of one sample, with NaN in unobserved slots."""
    window = data.sequence.input_window
    frames, valids, _, _ = load_input_frames(archive, sample, data)
    lags = [np.where(valid, frame[0], np.nan) for frame, valid in zip(frames, valids, strict=True)]
    shape = lags[0].shape
    ages = [np.full(shape, days, dtype=np.float32) for days in sample.days_before_target]
    empty = np.full(shape, np.nan, dtype=np.float32)
    pad = [empty] * (window - len(lags))
    columns = pad + lags + pad + ages
    day_of_year = sample.target_date.timetuple().tm_yday
    columns.append(np.full(shape, float(sample.lead_days), dtype=np.float32))
    columns.append(np.full(shape, np.sin(2.0 * np.pi * day_of_year / 365.25), dtype=np.float32))
    columns.append(np.full(shape, np.cos(2.0 * np.pi * day_of_year / 365.25), dtype=np.float32))
    return np.stack([column.reshape(-1) for column in columns], axis=1), shape


def fit_water(
    archive: SceneArchive, samples: list[SequenceSample], config: TabularBaselineConfig, water_mask: np.ndarray
) -> tuple[Regressor, int]:
    """Fit one water's model; return it with the number of candidate rows.

    At most ``max_train_rows`` rows are drawn uniformly over the adaptation set.
    """
    data = config.data
    rng = np.random.default_rng(config.seed)
    features, targets = [], []
    for sample in samples:
        matrix, _ = _sample_features(archive, sample, data)
        target_frame, _, _, target_valid = load_model_frames(archive.scenes[sample.target_index], data)
        usable = (target_valid & water_mask).reshape(-1)
        index = np.flatnonzero(usable)
        if index.size == 0:
            continue
        if index.size > config.pixels_per_sample:
            index = rng.choice(index, config.pixels_per_sample, replace=False)
        features.append(matrix[index])
        targets.append(target_frame[0].reshape(-1)[index])
    if not features:
        raise ValueError(f"No usable training pixels for {archive.water_id}")
    features, targets = np.concatenate(features), np.concatenate(targets)
    available = len(targets)
    if available > config.max_train_rows:
        # Samples run in time order, so stopping at the budget would drop the
        # most recent part of the adaptation set.
        keep = np.sort(rng.choice(available, config.max_train_rows, replace=False))
        features, targets = features[keep], targets[keep]
    print(f"{archive.water_id}: fitting on {len(targets)} of {available} candidate rows", flush=True)
    if config.estimator == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.max_iter,
            max_depth=config.max_depth,
            random_state=config.seed,
            n_jobs=config.n_jobs,
        )
    else:
        model = XGBRegressor(
            n_estimators=config.max_iter,
            learning_rate=config.learning_rate,
            objective="reg:squarederror",
            tree_method="hist",
            device=config.device,
            grow_policy="lossguide",
            max_leaves=31,
            max_depth=config.max_depth or 0,
            min_child_weight=20,
            max_bin=255,
            reg_lambda=0,
            reg_alpha=0,
            subsample=1,
            colsample_bytree=1,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        )
    # Both estimators accept NaN, so no imputation is needed.
    model.fit(features, targets)
    return model, available


def predict_sample(model: Regressor, archive: SceneArchive, sample: SequenceSample, data: DataConfig) -> np.ndarray:
    """``[1, H, W]`` log10 Chl-a forecast of one sample, NaN where no input slot is observed."""
    window = data.sequence.input_window
    matrix, shape = _sample_features(archive, sample, data)
    prediction = model.predict(matrix).astype(np.float32).reshape(1, *shape)
    observed = np.isfinite(matrix[:, :window]).any(axis=1).reshape(shape)
    prediction[0] = np.where(observed, prediction[0], np.nan)
    return prediction


def run_tabular_baseline(config_path: Path) -> PredictionOutput:
    config = TabularBaselineConfig.from_yaml(config_path)
    data = config.data
    run_dir = config.output_dir / config.run_name
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    archives, splits = load_waters(data)
    water_masks = water_extent_masks(archives, config)
    channel = PRODUCT_CHANNELS[data.forecast_product]
    rows = []
    for archive in archives:
        water_splits = splits[archive.water_id]
        model, available_rows = fit_water(archive, water_splits["train"], config, water_masks[archive.water_id])
        if config.estimator == "xgboost":
            model_dir = run_dir / "models"
            model_dir.mkdir(exist_ok=True)
            model.get_booster().save_model(model_dir / f"{archive.water_id}.json")
            provenance = {
                "config": json.loads(config.model_dump_json()),
                "xgboost_version": xgboost.__version__,
                "parameters": model.get_params(),
                "feature_count": model.n_features_in_,
                "training_sequences": len(water_splits["train"]),
                "training_rows": min(available_rows, config.max_train_rows),
                "available_rows": available_rows,
                "feature_recipe": "lagged log10 Chl-a, slot ages, lead days, day-of-year sin/cos",
            }
            (model_dir / f"{archive.water_id}_fit.json").write_text(json.dumps(provenance, indent=2))
            # The model fitted on the GPU predicts on CPU arrays.
            model.set_params(device="cpu")
        for sample in water_splits[config.split]:
            target_scene = archive.scenes[sample.target_index]
            prediction_path = prediction_dir / f"{sample_id(sample)}.tif"
            write_prediction_raster(
                prediction_path, predict_sample(model, archive, sample, data), target_scene, [channel]
            )
            rows.append(
                prediction_row(
                    sample,
                    target_scene,
                    prediction_path,
                    [channel],
                    split=config.split,
                    model_name=f"{config.estimator}_pixel",
                    checkpoint=None,
                )
            )
    manifest_path = run_dir / "prediction_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    return PredictionOutput(run_dir=run_dir, manifest_path=manifest_path, prediction_count=len(rows))
