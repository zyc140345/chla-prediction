from __future__ import annotations

import numpy as np

from chla_prediction.evaluation.bootstrap import mean_ci


def test_mean_ci_brackets_the_sample_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    low, high = mean_ci(values, rng=np.random.default_rng(7), n_resamples=200)

    assert low <= values.mean() <= high


def test_mean_ci_ignores_non_finite_samples() -> None:
    with_nan = np.array([1.0, 2.0, np.nan, 4.0])
    without = np.array([1.0, 2.0, 4.0])

    assert mean_ci(with_nan, rng=np.random.default_rng(3), n_resamples=200) == mean_ci(
        without, rng=np.random.default_rng(3), n_resamples=200
    )


def test_mean_ci_is_undefined_below_two_samples() -> None:
    low, high = mean_ci(np.array([1.0]), rng=np.random.default_rng(0), n_resamples=200)

    assert np.isnan(low)
    assert np.isnan(high)
