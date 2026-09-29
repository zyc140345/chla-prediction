"""Tables of the data-amount study: learning curves over seeds and the training trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from chla_prediction.io import read_jsonl
from chla_prediction.recipe import EVALUATION_WATERS

__all__ = ["summarize_learning_curves"]

MODEL_PATTERN = re.compile(r"^curve_ours_(scratch|adapted)_(1y|2y|all)_s(42|43|44)$")
RUN_PATTERN = re.compile(r"^train_(curve_ours_(scratch|adapted)_(1y|2y|all)_s(42|43|44))_(.+)$")
REGIME_ORDER = ["scratch", "adapted"]
AMOUNT_ORDER = ["1y", "2y", "all"]


def _decode_model(model: str) -> tuple[str, str, int] | None:
    match = MODEL_PATTERN.fullmatch(model)
    if match is None:
        return None
    regime, amount, seed = match.groups()
    return regime, amount, int(seed)


def _sort_runs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["regime"] = pd.Categorical(frame["regime"], REGIME_ORDER, ordered=True)
    frame["data_amount"] = pd.Categorical(frame["data_amount"], AMOUNT_ORDER, ordered=True)
    frame = frame.sort_values(["regime", "data_amount", "seed"]).reset_index(drop=True)
    frame["regime"] = frame["regime"].astype("string")
    frame["data_amount"] = frame["data_amount"].astype("string")
    return frame


def _seed_metrics(evaluation_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(evaluation_dir / "overall_summary_metrics.csv")
    rows = []
    for row in source.to_dict(orient="records"):
        decoded = _decode_model(str(row["model"]))
        if decoded is None:
            continue
        regime, amount, seed = decoded
        rows.append({"regime": regime, "data_amount": amount, "seed": seed, **row})
    if not rows:
        raise ValueError(f"No learning-curve models found in {evaluation_dir}")
    return _sort_runs(pd.DataFrame(rows))


def _aggregate_seeds(seeds: pd.DataFrame) -> pd.DataFrame:
    excluded = {"regime", "data_amount", "seed", "model", "n_waters"}
    metrics = [column for column in seeds.columns if column not in excluded]
    grouped = seeds.groupby(["regime", "data_amount"], observed=True, sort=False)
    summary = grouped.size().rename("n_seeds").to_frame()
    for metric in metrics:
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_std"] = grouped[metric].std(ddof=1)
    return _sort_runs(summary.reset_index().assign(seed=0)).drop(columns="seed")


def _training_trajectories(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step and epoch logs of the data-amount runs on the evaluation waters."""
    step_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    for run_dir in sorted(root.glob("train_curve_ours_*")):
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if match is None:
            continue
        model, regime, amount, seed, water_id = match.groups()
        if water_id not in EVALUATION_WATERS:
            continue
        manifest = json.loads((run_dir / "pretrain_manifest.json").read_text())
        counts = manifest["sample_counts"]
        common = {
            "model": model,
            "regime": regime,
            "data_amount": amount,
            "seed": int(seed),
            "water_id": water_id,
            "train_sequences": counts["train_sequences"],
            "val_sequences": counts["val_sequences"],
            "source_revision": manifest["git_hash"],
        }
        steps = read_jsonl(run_dir / "training_steps.jsonl")
        step_rows += [{**common, "global_step": step, **row} for step, row in enumerate(steps, start=1)]
        epoch_rows += [{**common, **row} for row in read_jsonl(run_dir / "training_log.jsonl")]
    if not step_rows or not epoch_rows:
        raise ValueError(f"No complete learning-curve training runs found under {root}")
    return pd.DataFrame(step_rows), pd.DataFrame(epoch_rows)


def summarize_learning_curves(root: Path, evaluation_dir: Path, output_dir: Path) -> None:
    """Write per-seed and seed-averaged learning-curve tables and the training trajectories."""
    seeds = _seed_metrics(evaluation_dir)
    summary = _aggregate_seeds(seeds)
    steps, epochs = _training_trajectories(root)

    output_dir.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(output_dir / "learning_curve_seeds.csv", index=False)
    summary.to_csv(output_dir / "learning_curve_summary.csv", index=False)
    steps.to_csv(output_dir / "learning_curve_training_steps.csv", index=False)
    epochs.to_csv(output_dir / "learning_curve_epochs.csv", index=False)
