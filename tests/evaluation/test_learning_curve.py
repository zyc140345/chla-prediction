import json

import pandas as pd
import pytest

from chla_prediction.evaluation.learning_curve import summarize_learning_curves
from chla_prediction.io import write_jsonl


def test_learning_curve_summary_writes_seed_metrics_and_training_trajectories(tmp_path) -> None:
    root = tmp_path / "forecast"
    evaluation = root / "evaluation_learning_curve"
    evaluation.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model": f"curve_ours_{regime}_{amount}_s{seed}",
                "rmse_log": value,
                "chla_rmse": value * 10,
                "n_waters": 5,
            }
            for regime, amount, seed, value in [
                ("scratch", "1y", 42, 0.60),
                ("scratch", "1y", 43, 0.70),
                ("adapted", "1y", 42, 0.40),
                ("adapted", "1y", 43, 0.50),
            ]
        ]
    ).to_csv(evaluation / "overall_summary_metrics.csv", index=False)
    run = root / "train_curve_ours_adapted_1y_s42_hushan"
    run.mkdir()
    write_jsonl(
        run / "training_steps.jsonl",
        [
            {"phase": "warmup", "epoch": 0, "step": 1, "loss": 0.8},
            {"phase": "forecast", "epoch": 0, "step": 1, "loss": 0.5},
        ],
    )
    write_jsonl(run / "training_log.jsonl", [{"phase": "forecast", "epoch": 0, "train_loss": 0.5, "val_loss": 0.6}])
    manifest = {"git_hash": "abc123", "sample_counts": {"train_sequences": 10, "val_sequences": 2}}
    (run / "pretrain_manifest.json").write_text(json.dumps(manifest))

    output = root / "paper"
    summarize_learning_curves(root, evaluation, output)

    seeds = pd.read_csv(output / "learning_curve_seeds.csv")
    assert seeds[["regime", "data_amount", "seed"]].values.tolist() == [
        ["scratch", "1y", 42],
        ["scratch", "1y", 43],
        ["adapted", "1y", 42],
        ["adapted", "1y", 43],
    ]
    summary = pd.read_csv(output / "learning_curve_summary.csv")
    scratch = summary.query("regime == 'scratch'").iloc[0]
    assert scratch["n_seeds"] == 2
    assert scratch["rmse_log_mean"] == pytest.approx(0.65)
    assert scratch["rmse_log_std"] == pytest.approx(0.070711, abs=1e-6)
    steps = pd.read_csv(output / "learning_curve_training_steps.csv")
    assert steps["global_step"].tolist() == [1, 2]
    assert steps["water_id"].tolist() == ["hushan", "hushan"]
    assert steps["regime"].tolist() == ["adapted", "adapted"]
    epochs = pd.read_csv(output / "learning_curve_epochs.csv")
    assert epochs.loc[0, "val_loss"] == 0.6
