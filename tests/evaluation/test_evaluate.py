import numpy as np
import pandas as pd

from chla_prediction.evaluation.evaluate import _horizon_summaries, horizon_bins
from chla_prediction.evaluation.scoring import CHLA_METRICS, OUTPUT_METRICS


def test_horizon_bins_label_every_lead() -> None:
    bins = horizon_bins(np.array([1, 7, 8, 15, 16, 30, 31, 485]), [7, 15, 30])
    assert list(bins) == ["<=7d", "<=7d", "8-15d", "8-15d", "16-30d", "16-30d", ">30d", ">30d"]
    assert list(bins.categories) == ["<=7d", "8-15d", "16-30d", ">30d"]


def test_horizon_summaries_score_each_bin_separately() -> None:
    rows = []
    for model, error in [("persistence", 0.5), ("ours", 0.3)]:
        for index, lead in enumerate([3, 5, 20, 40]):
            row = {"model": model, "sample_id": f"s{index}", "water_id": "water", "horizon_days": lead}
            rows.append(row | dict.fromkeys(OUTPUT_METRICS + CHLA_METRICS, error))
    per_sample = pd.DataFrame(rows)
    per_sample["horizon_bin"] = horizon_bins(per_sample.horizon_days.to_numpy(), [7, 15, 30])

    pooled, overall = _horizon_summaries(per_sample, "persistence")
    assert list(pooled.horizon_bin.unique()) == ["<=7d", "16-30d", ">30d"]
    assert pooled[pooled.model == "ours"].n_samples.tolist() == [2, 1, 1]
    # Every bin keeps the pooled scoring, so the paired difference is the same.
    assert np.allclose(pooled[pooled.model == "ours"].rmse_log_diff_vs_reference, -0.2)
    assert overall.n_waters.unique().tolist() == [1]
