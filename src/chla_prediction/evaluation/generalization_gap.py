"""Generalization gap (test minus adaptation-set error) of the baselines trained on the target water alone.

Each task of a plan scores one baseline on one water's adaptation set
(``train`` split) and test set; its test scores must match the main
evaluation before the train scores are kept. ``summarize_generalization_gap``
adds the proposed model and writes the gap table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from xgboost import XGBRegressor

from chla_prediction.baselines.tabular import Regressor, fit_water, predict_sample
from chla_prediction.config import (
    PRODUCT_CHANNELS,
    EvaluateForecastConfig,
    TabularBaselineConfig,
    TrainForecasterConfig,
)
from chla_prediction.evaluation.diagnostics import CheckpointEvaluator
from chla_prediction.evaluation.scoring import SampleScorer, score_split, write_scores
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.waters import load_waters
from chla_prediction.io import file_sha256

__all__ = ["METHOD_LABELS", "check_test_scores", "gap_tables", "score_task", "summarize_generalization_gap"]

METRICS = ["rmse_log", "chla_rmse", "chla_mae", "chla_pearson_r"]
# Methods of the gap table, in print order.
METHOD_LABELS = {
    "rf_pixel": "Random forest",
    "xgboost_pixel": "XGBoost",
    "scratch_convlstm": "ConvLSTM",
    "scratch_ours": "Ours (from scratch)",
    "ours_adapted": "Ours (pretrained + adapted)",
}


def check_test_scores(scored: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Require the scene and pixel coverage of the main evaluation and agreement within float precision."""
    old = reference.sort_values("sample_id").reset_index(drop=True)
    new = scored.sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        old[["sample_id", "water_id", "n_valid_pixels"]],
        new[["sample_id", "water_id", "n_valid_pixels"]],
    )
    differences = {}
    for metric in METRICS:
        # Float32 convolutions and retrievals differ slightly between GPU models.
        atol = 1e-3 if metric in ("chla_rmse", "chla_mae") else 1e-5
        np.testing.assert_allclose(new[metric], old[metric], rtol=1e-5, atol=atol, equal_nan=True)
        differences[metric] = float((new[metric] - old[metric]).abs().max())
    return {
        "test_pairs": len(new),
        "max_absolute_differences": differences,
        "rtol": 1e-5,
        "log_and_correlation_atol": 1e-5,
        "linear_error_atol": 1e-3,
    }


def score_tree(
    model: Regressor,
    config: TabularBaselineConfig,
    evaluation: EvaluateForecastConfig,
    model_path: Path,
    output: Path,
    split: str,
) -> pd.DataFrame:
    """Score one split with next-observation targets, even if the fit used ``train_max_lead_days``."""
    data = config.data.next_observation_only()
    archives, splits = load_waters(data)
    frame, candidates = score_split(
        lambda archive, sample: predict_sample(model, archive, sample, data),
        archives,
        splits,
        split,
        [PRODUCT_CHANNELS["mdn_chla"]],
        SampleScorer(evaluation),
        water_extent_masks(archives, evaluation),
        evaluation.min_valid_fraction,
    )
    if len(frame):
        frame = frame.assign(split=split, checkpoint=str(model_path))
    write_scores(
        frame,
        output,
        checkpoint=str(model_path),
        split=split,
        candidate_sequences=candidates,
        scored_sequences=len(frame),
        task="next_observation",
    )
    return frame


def score_task(task: dict, plan: dict) -> None:
    """Score one baseline on one water; the test scores must match the main evaluation first."""
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    destination = Path(task["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "completed.json").exists():
        raise FileExistsError(f"Completed scores are immutable: {destination}")
    reference = pd.read_csv(plan["canonical"])
    reference = reference[(reference.model == task["variant"]) & (reference.water_id == task["water_id"])]
    evaluation = EvaluateForecastConfig.model_validate(task["evaluation_config"])
    if task["variant"] == "scratch_convlstm":
        config = TrainForecasterConfig.model_validate(task["train_config"])
        evaluator = CheckpointEvaluator(config, evaluation)
        model_path = Path(task["checkpoint"])
        check = check_test_scores(evaluator.score(model_path, "test", destination / "test.csv"), reference)
        evaluator.score(model_path, "train", destination / "train.csv")
    else:
        config = TabularBaselineConfig.model_validate(task["train_config"])
        config = config.model_copy(update={"n_jobs": threads})
        if task["variant"] == "rf_pixel":
            # Random forests are not saved by the main run, so the fit is repeated.
            archives, splits = load_waters(config.data)
            (archive,) = archives
            water_mask = water_extent_masks(archives, config)[archive.water_id]
            model, _ = fit_water(archive, splits[archive.water_id]["train"], config, water_mask)
            model_path = destination / "random_forest.joblib"
            joblib.dump(model, model_path, compress=3)
        else:
            model_path = Path(task["checkpoint"])
            model = XGBRegressor()
            model.load_model(model_path)
            model.set_params(device="cpu", n_jobs=threads)
        test = score_tree(model, config, evaluation, model_path, destination / "test.csv", "test")
        check = check_test_scores(test, reference)
        score_tree(model, config, evaluation, model_path, destination / "train.csv", "train")
    (destination / "completed.json").write_text(
        json.dumps(
            {
                "variant": task["variant"],
                "water_id": task["water_id"],
                "seed": 42,
                "data_amount": "all",
                "checkpoint": str(model_path),
                "checkpoint_sha256": file_sha256(model_path),
                "test_verification": check,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "sklearn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "torch_version": torch.__version__,
                "training_config": json.loads(config.model_dump_json()),
            },
            indent=2,
        )
        + "\n"
    )


def gap_tables(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-water gaps and their mean over waters; scenes are averaged within each water first."""
    groups = samples.groupby(["model", "water_id", "split"])
    per_water = groups[METRICS].mean().join(groups.size().rename("n_samples")).reset_index()
    gaps = per_water.pivot(index=["model", "water_id"], columns="split", values="rmse_log")
    gaps["test_minus_train"] = gaps["test"] - gaps["train"]
    counts = per_water.pivot(index=["model", "water_id"], columns="split", values="n_samples")
    gaps = gaps.join(counts.add_prefix("n_")).reset_index()
    if gaps[["train", "test"]].isna().any().any():
        raise ValueError("Every method needs train and test scores on every water")
    rows = [
        {
            "model": model,
            "seed": 42,
            "data_amount": "all",
            "n_waters": len(group),
            "train_rmse_log": group.train.mean(),
            "test_rmse_log": group.test.mean(),
            "test_minus_train": group.test_minus_train.mean(),
            "n_train_pairs": int(group.n_train.sum()),
            "n_test_pairs": int(group.n_test.sum()),
        }
        for model, group in gaps.groupby("model", sort=False)
    ]
    return gaps, pd.DataFrame(rows)


def summarize_generalization_gap(stage: Path) -> None:
    """Join the scored tasks of a plan directory with the proposed model's scores into the gap table."""
    plan = json.loads((stage / "plan.json").read_text())
    reference = pd.read_csv(plan["canonical"])
    frames, coverage = [], []
    for task in plan["tasks"]:
        destination = Path(task["destination"])
        completed = json.loads((destination / "completed.json").read_text())
        if (completed["variant"], completed["water_id"]) != (task["variant"], task["water_id"]):
            raise ValueError(f"{destination} holds another task")
        frames.append(pd.read_csv(destination / "train.csv").assign(model=task["variant"], seed=42, data_amount="all"))
        coverage += [
            json.loads((destination / f"{split}.json").read_text())
            | {"model": task["variant"], "water_id": task["water_id"]}
            for split in ("train", "test")
        ]
    curve = pd.read_csv(stage.parent / "overfit_diagnostics/paper/curve_per_sample.csv")
    curve = curve[(curve.seed == 42) & (curve.data_amount == "all") & (curve.split == "train")].copy()
    curve["model"] = curve.regime.map({"scratch": "scratch_ours", "adapted": "ours_adapted"})
    frames.append(curve)
    # Test scores come from the main evaluation; rescoring on another GPU matches only to float precision.
    frames.append(reference[reference.model.isin(METHOD_LABELS)].assign(split="test", seed=42, data_amount="all"))
    samples = pd.concat(frames, ignore_index=True)
    if samples.duplicated(["model", "sample_id", "split"]).any() or set(samples.model) != set(METHOD_LABELS):
        raise ValueError("Scores must cover each method of METHOD_LABELS once per sample and split")
    gaps, summary = gap_tables(samples)
    if not ((summary.n_waters == 5).all() and summary.n_test_pairs.nunique() == 1):
        raise ValueError("Every method must be scored on the same test pairs of all five waters")
    summary["method"] = summary.model.map(METHOD_LABELS)
    summary = summary.set_index("model").loc[list(METHOD_LABELS)].reset_index()
    samples.to_csv(stage / "per_sample_metrics.csv", index=False)
    gaps.to_csv(stage / "per_water_gaps.csv", index=False)
    summary.to_csv(stage / "generalization_gap.csv", index=False)
    pd.DataFrame(coverage).to_csv(stage / "baseline_coverage.csv", index=False)
    print(summary.to_string(index=False), flush=True)
