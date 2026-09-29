import numpy as np
import pandas as pd

from chla_prediction.evaluation import paper_tables


def test_claims_pair_samples_and_count_the_waters_that_agree() -> None:
    errors = {"ours_adapted": [0.20, 0.30], "notime_adapted": [0.40, 0.25]}
    per_sample = pd.DataFrame(
        [
            {"sample_id": sample_id, "water_id": water_id, "model": model, "rmse_log": error}
            for model, model_errors in errors.items()
            for sample_id, water_id, error in zip(("s1", "s2"), ("water_a", "water_b"), model_errors, strict=True)
        ]
    )
    wide = per_sample.pivot_table(index="sample_id", columns="model", values="rmse_log")
    water_of = per_sample.drop_duplicates("sample_id").set_index("sample_id").water_id.reindex(wide.index)

    table = paper_tables.claims_table(wide, water_of, ["water_a", "water_b"])

    claim = table.set_index("against").loc["notime_adapted"]
    assert np.isclose(claim.mean_diff, -0.075)
    assert np.isclose(claim.diff_water_a, -0.20)
    assert np.isclose(claim.diff_water_b, 0.05)
    assert claim.waters_agreeing == "1/2"
    assert claim.n_samples == 2


def test_main_table_reports_spatial_pearson_correlation() -> None:
    columns = ["sample_id", "water_id", "model", "rmse_log", "chla_rmse", "chla_mae", "chla_pearson_r"]
    per_sample = pd.DataFrame(
        [
            ("s1", "water_a", "ours", 0.2, 2.0, 1.0, 0.8),
            ("s2", "water_b", "ours", 0.3, 3.0, 2.0, 0.6),
            ("s1", "water_a", "persistence", 0.4, 4.0, 3.0, 0.5),
            ("s2", "water_b", "persistence", 0.5, 5.0, 4.0, 0.3),
        ],
        columns=columns,
    )
    wide = per_sample.pivot_table(index="sample_id", columns="model", values="rmse_log")

    table = paper_tables.main_table(
        per_sample, wide, ["water_a", "water_b"], [("ours", "Ours", "Pretrained + adapted")]
    )

    assert np.isclose(table.iloc[0].pearson_r, 0.7)
