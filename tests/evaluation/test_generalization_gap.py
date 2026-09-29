import pandas as pd
import pytest

from chla_prediction.evaluation.generalization_gap import check_test_scores, gap_tables


@pytest.fixture
def expected_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["first", "second"],
            "water_id": ["water", "water"],
            "n_valid_pixels": [100, 200],
            "rmse_log": [0.3, 0.4],
            "chla_rmse": [20.0, 30.0],
            "chla_mae": [10.0, 15.0],
            "chla_pearson_r": [0.2, 0.3],
        }
    )


def test_check_test_scores_pairs_scenes_and_rejects_changed_pixel_counts(expected_scores) -> None:
    observed = expected_scores.iloc[::-1].copy()
    observed["rmse_log"] += 1e-7
    assert check_test_scores(observed, expected_scores)["test_pairs"] == 2
    observed.loc[0, "n_valid_pixels"] -= 1
    with pytest.raises(AssertionError):
        check_test_scores(observed, expected_scores)


def test_check_test_scores_rejects_scene_errors_that_drift_while_the_mean_holds(expected_scores) -> None:
    observed = expected_scores.copy()
    observed["rmse_log"] += [0.02, -0.02]
    with pytest.raises(AssertionError):
        check_test_scores(observed, expected_scores)


def test_gap_table_weights_waters_equally_instead_of_pooling_scenes() -> None:
    rows = []
    for water, count, train, test in [("small", 1, 0.2, 0.3), ("large", 3, 0.6, 0.8)]:
        for split, error in [("train", train), ("test", test)]:
            row = {"model": "baseline", "water_id": water, "split": split, "rmse_log": error}
            rows += [row | {"chla_rmse": 20, "chla_mae": 10, "chla_pearson_r": 0.3}] * count
    _, summary = gap_tables(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row.train_rmse_log == pytest.approx(0.4)
    assert row.test_rmse_log == pytest.approx(0.55)
    assert row.test_minus_train == pytest.approx(0.15)
    assert row.n_train_pairs == row.n_test_pairs == 4
