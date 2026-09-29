import json

import pandas as pd
import pytest

from chla_prediction.evaluation.diagnostics_summary import select_matched, summarize_diagnostics


def test_matched_selection_compares_both_learning_rates_on_validation_only(tmp_path) -> None:
    tasks = []
    for rate, error in [(0.00005, 0.4), (0.0002, 0.3)]:
        folder = tmp_path / str(rate)
        folder.mkdir()
        (folder / "completed.json").write_text("{}")
        pd.DataFrame(
            [
                {
                    "split": "val",
                    "epoch": 3,
                    "learning_rate": rate,
                    "rmse_log": error,
                    "checkpoint": f"{folder}/checkpoint.pt",
                    "n_samples": 3,
                },
                {
                    "split": "train",
                    "epoch": 40,
                    "learning_rate": rate,
                    "rmse_log": 0.0,
                    "checkpoint": f"{folder}/unselected.pt",
                    "n_samples": 10,
                },
            ]
        ).to_csv(folder / "trajectory.csv", index=False)
        tasks.append(
            {
                "destination": str(folder),
                "regime": "adapted",
                "data_amount": "1y",
                "seed": 42,
                "water_id": "hushan",
                "train_config": f"{rate}.yaml",
                "evaluation_config": "eval.yaml",
            }
        )
    select_matched(tasks, tmp_path / "paper")
    selected = json.loads((tmp_path / "paper/diagnostics_selected.jsonl").read_text())
    assert selected["learning_rate"] == 0.0002
    assert selected["epoch"] == 3
    assert selected["kind"] == "selected"


def test_summary_averages_waters_and_reports_missing_validation(tmp_path) -> None:
    tasks = []
    for regime in ("scratch", "adapted"):
        for water, n_samples, error in [("small", 1, 0.2), ("large", 3, 0.6)]:
            root = tmp_path / f"{regime}_{water}"
            root.mkdir()
            (root / "completed.json").write_text("{}")
            tasks.append(
                {"destination": str(root), "regime": regime, "data_amount": "all", "seed": 42, "water_id": water}
            )
            for split in ("train", "val", "test"):
                n = 0 if water == "small" and split == "val" else n_samples
                (root / f"{split}.json").write_text(json.dumps({"split": split, "scored_sequences": n}))
                if n:
                    pd.DataFrame(
                        [
                            {
                                "sample_id": f"{water}_{i}",
                                "split": split,
                                "n_valid_pixels": 10,
                                "rmse_log": error,
                                "chla_rmse": error,
                                "chla_mae": error,
                                "chla_pearson_r": 0.5,
                            }
                            for i in range(n)
                        ]
                    ).to_csv(root / f"{split}.csv", index=False)
    summarize_diagnostics(tasks, tmp_path / "paper", "curve")
    overall = pd.read_csv(tmp_path / "paper/curve_overall_seeds.csv")
    assert overall[overall.split == "test"].rmse_log.tolist() == pytest.approx([0.4, 0.4])
    assert overall[overall.split == "val"].n_waters.tolist() == [1, 1]
    gaps = pd.read_csv(tmp_path / "paper/curve_gaps_per_water.csv")
    assert gaps[gaps.water_id == "small"].val_minus_train.isna().all()
