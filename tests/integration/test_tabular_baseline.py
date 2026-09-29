"""The per-pixel regressors, fitted per water on a synthetic twelve-scene archive."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from xgboost import XGBRegressor

from chla_prediction.baselines import tabular
from chla_prediction.config import TabularBaselineConfig
from chla_prediction.experiments.evaluate_forecast import main as evaluate_main
from chla_prediction.experiments.run_tabular_baseline import main as tabular_main
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.waters import load_waters
from chla_prediction.io import read_jsonl

SEQUENCE = {
    "input_window": 3,
    "min_input_observations": 2,
    "train_end": "2021-03-02",
    "val_end": "2021-03-22",
    "train_max_lead_days": 30,
}


@pytest.fixture(scope="module")
def archive(tmp_path_factory, write_scene_archive):
    """A synthetic twelve-scene archive that the tests only read."""
    return write_scene_archive(tmp_path_factory.mktemp("archive"), n_scenes=12, seed=11)


@pytest.fixture
def tabular_config(tmp_path, archive):
    """A per-pixel baseline config on NDCI Chl-a inputs; ``fields`` override the top level."""

    def config(run_name: str, **fields) -> dict:
        return {
            "run_name": run_name,
            "output_dir": str(tmp_path / "outputs"),
            "data": {"archives": archive.archives, "sequence": SEQUENCE, "input_product": "ndci_chla"},
            "split": "test",
            "pixels_per_sample": 64,
            "water_mask_dir": str(archive.water_mask_dir),
        } | fields

    return config


def test_tabular_baseline_enters_the_shared_evaluation(tmp_path, archive, tabular_config, run_experiment) -> None:
    """The regressor is fitted per water and scored through the same manifest contract as a network."""
    run_experiment(tabular_main, tmp_path / "xgboost.yaml", tabular_config("xgboost", max_iter=20))

    manifest_path = tmp_path / "outputs" / "xgboost" / "prediction_manifest.jsonl"
    rows = read_jsonl(manifest_path)
    assert (tmp_path / "outputs" / "xgboost" / "models" / "hushan.json").is_file()
    assert all(row["model_name"] == "xgboost_pixel" for row in rows)
    assert rows
    assert all(row["bands"] == ["LOG10_CHLA_NDCI"] for row in rows)
    with rasterio.open(rows[0]["prediction_path"]) as src:
        prediction = src.read(1)
    assert np.isfinite(prediction).any()
    assert np.nanmax(prediction) < 3.0

    evaluate = {
        "run_name": "evaluation",
        "output_dir": str(tmp_path / "outputs"),
        "data": {"archives": archive.archives, "sequence": SEQUENCE},
        "prediction_manifests": {
            "xgboost_pixel": str(manifest_path),
            "missing_variant": str(tmp_path / "absent.jsonl"),
        },
        "skill_reference": "xgboost_pixel",
        "chla_axis": "ndci",
        "water_mask_dir": str(archive.water_mask_dir),
        "min_valid_fraction": 0.01,
        "horizon_bins": [10, 20],
    }
    run_experiment(evaluate_main, tmp_path / "evaluate.yaml", evaluate)
    summary = pd.read_csv(tmp_path / "outputs" / "evaluation" / "summary_metrics.csv")
    assert summary.model.tolist() == ["xgboost_pixel"]
    assert summary.rmse_log.notna().all()
    by_horizon = pd.read_csv(tmp_path / "outputs" / "evaluation" / "overall_summary_by_horizon.csv")
    assert set(by_horizon.horizon_bin) <= {"<=10d", "11-20d", ">20d"}


def test_random_forest_variant_records_its_own_estimator(tmp_path, tabular_config, run_experiment) -> None:
    """The random forest takes the unobserved-slot NaN in fit and in predict alike.

    The manifest names it as the estimator that ran.
    """
    config = tabular_config("rf", estimator="random_forest", max_iter=5, max_depth=4)
    run_experiment(tabular_main, tmp_path / "rf.yaml", config)

    rows = read_jsonl(tmp_path / "outputs" / "rf" / "prediction_manifest.jsonl")
    assert rows
    assert all(row["model_name"] == "random_forest_pixel" for row in rows)
    with rasterio.open(rows[0]["prediction_path"]) as src:
        prediction = src.read(1)
    assert np.isfinite(prediction).any()


def test_row_budget_draws_over_the_whole_adaptation_set(tabular_config, monkeypatch) -> None:
    """Samples run in time order, so a budget below the candidate rows must still keep rows of the latest samples."""
    config = TabularBaselineConfig.model_validate(tabular_config("xgboost", max_iter=2))
    archives, splits = load_waters(config.data)
    samples = splits["hushan"]["train"]
    water = water_extent_masks(archives, config)["hushan"]
    fitted = {}
    monkeypatch.setattr(XGBRegressor, "fit", lambda self, features, targets: fitted.update(features=features))
    _, available = tabular.fit_water(archives[0], samples, config, water)
    assert len(fitted["features"]) == available

    budget = available // 2
    tabular.fit_water(archives[0], samples, config.model_copy(update={"max_train_rows": budget}), water)
    assert len(fitted["features"]) == budget
    # The last two feature columns are the target date's day-of-year sine and cosine.
    latest = samples[-1].target_date.timetuple().tm_yday
    season = np.array([np.sin(2 * np.pi * latest / 365.25), np.cos(2 * np.pi * latest / 365.25)])
    assert np.isclose(fitted["features"][:, -2:], season, atol=1e-5).all(axis=1).any()
