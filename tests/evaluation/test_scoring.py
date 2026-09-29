import numpy as np
import pytest
import rasterio

from chla_prediction.config import EvaluateForecastConfig
from chla_prediction.evaluation.scoring import SampleScorer, chla_metrics
from chla_prediction.retrieval.empirical import three_band_chla


@pytest.fixture
def scoring_case(tmp_path, write_raster, make_data_config):
    """A scorer and the manifest row of a prediction and its reference scene, which is clear everywhere."""

    def case(
        reference: np.ndarray, reference_bands: list[str], prediction: np.ndarray, bands: list[str], chla_axis: str
    ) -> tuple[SampleScorer, dict]:
        scl = np.full((1, *reference.shape[1:]), 6, np.uint8)
        row = {
            "sample_id": "sample",
            "water_id": "water",
            "origin_date": "20240701",
            "target_date": "20240706",
            "horizon_days": 5,
            "bands": bands,
            "prediction_path": str(write_raster(tmp_path / "prediction.tif", prediction)),
            "reference_path": str(write_raster(tmp_path / "reference.tif", reference, reference_bands)),
            "reference_scl_path": str(write_raster(tmp_path / "scl.tif", scl)),
        }
        config = EvaluateForecastConfig(
            run_name="test",
            data=make_data_config(),
            prediction_manifests={"model": "unused.jsonl"},
            chla_axis=chla_axis,
        )
        return SampleScorer(config), row

    return case


def test_chla_pearson_is_computed_in_linear_space() -> None:
    prediction = np.array([1.0, 2.0, 4.0])
    reference = np.array([1.0, 3.0, 9.0])
    metrics = chla_metrics(prediction, reference, (0.1, 1000.0))
    assert np.isclose(metrics["chla_pearson_r"], np.corrcoef(prediction, reference)[0, 1])


def test_chla_pearson_is_undefined_for_a_constant_field() -> None:
    metrics = chla_metrics(np.ones(3), np.arange(1.0, 4.0), (0.1, 1000.0))
    assert np.isnan(metrics["chla_pearson_r"])


def test_rmse_log_is_the_log10_error_of_the_clipped_fields() -> None:
    """Forecasts a factor ten off score 1; values beyond the Chl-a range are clipped first."""
    metrics = chla_metrics(np.array([10.0, 100.0, 5000.0]), np.array([1.0, 10.0, 100.0]), (0.1, 1000.0))
    assert np.isclose(metrics["rmse_log"], 1.0)


def test_reflectance_nan_is_excluded_from_coverage_and_all_metrics(scoring_case) -> None:
    reference = np.array([[[1000, 1000, 1000, 1000]], [[1000, 2000, 1000, 1000]]], dtype=np.uint16)
    prediction = reference.astype(np.float32) * 1e-4
    prediction[:, 0, 2] = np.nan
    prediction[1, 0, 3] = np.nan
    scorer, row = scoring_case(reference, ["B4", "B5"], prediction, ["B4", "B5"], chla_axis="ndci")
    metrics = scorer.score(row, np.ones((1, 4), dtype=bool))
    assert metrics["n_valid_pixels"] == 2
    assert metrics["water_mask_fraction"] == 0.5
    assert metrics["chla_valid_fraction"] == 1.0
    for name in ("rmse", "mae", "bias", "ndci_mae", "chla_rmse", "chla_mae", "rmse_log"):
        assert metrics[name] == pytest.approx(0.0)
    with rasterio.open(row["prediction_path"], "r+") as stream:
        stream.write(np.full_like(prediction, np.nan))
    assert scorer.score(row, np.ones((1, 4), dtype=bool)) is None


def test_three_band_scorer_uses_own_reference_and_excludes_invalid_retrievals(scoring_case) -> None:
    bands = ["B4", "B5", "B6"]
    reference = np.array([[[100, 200, 100]], [[200, 100, 100]], [[100, 100, 100]]], dtype=np.uint16)
    concentration = three_band_chla(reference.astype(np.float32) * 1e-4, bands)
    prediction = np.log10(np.nan_to_num(concentration, nan=1))[None]
    scorer, row = scoring_case(reference, bands, prediction, ["LOG10_CHLA_3BDA"], chla_axis="three_band")
    metrics = scorer.score(row, np.ones((1, 3), bool))
    assert metrics["n_valid_pixels"] == 2
    assert metrics["water_mask_fraction"] == pytest.approx(0.6667)
    assert metrics["rmse_log"] == pytest.approx(0, abs=1e-6)
