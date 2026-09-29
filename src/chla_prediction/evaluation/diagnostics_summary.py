"""Aggregate completed diagnostics tasks and select matched checkpoints on validation only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chla_prediction.evaluation.diagnostics import METRIC, select_validation_checkpoint
from chla_prediction.evaluation.evaluate import overall_means, seed_summary
from chla_prediction.io import write_jsonl

__all__ = ["select_matched", "summarize_diagnostics"]

KEYS = ["regime", "data_amount", "seed", "water_id"]
METRICS = ["rmse_log", "chla_rmse", "chla_mae", "chla_pearson_r"]


def select_matched(tasks: list[dict], output: Path) -> None:
    """Pick each condition's checkpoint by validation error and write the plan that scores it."""
    trajectories = []
    for task in tasks:
        root = Path(task["destination"])
        if not (root / "completed.json").exists():
            raise ValueError(f"Unfinished diagnostic task: {root}")
        frame = pd.read_csv(root / "trajectory.csv")
        for key in KEYS:
            frame[key] = task[key]
        frame["train_config"] = task["train_config"]
        frame["evaluation_config"] = task["evaluation_config"]
        trajectories.append(frame)
    trajectory = pd.concat(trajectories, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output / "matched_trajectories.csv", index=False)
    selected, plan = [], []
    for key, candidates in trajectory.groupby(KEYS):
        winner = select_validation_checkpoint(candidates)
        metadata = dict(zip(KEYS, key, strict=True))
        metadata["seed"] = int(metadata["seed"])
        selected.append(
            metadata
            | {
                "epoch": int(winner.epoch),
                "learning_rate": float(winner.learning_rate),
                "validation_rmse_log": float(winner[METRIC]),
                "checkpoint": winner.checkpoint,
            }
        )
        stem = "_".join(map(str, key))
        plan.append(
            metadata
            | {
                "kind": "selected",
                "train_config": winner.train_config,
                "evaluation_config": winner.evaluation_config,
                "checkpoint": winner.checkpoint,
                "epoch": int(winner.epoch),
                "learning_rate": float(winner.learning_rate),
                "destination": str(output.parent / "selected" / stem),
            }
        )
    pd.DataFrame(selected).to_csv(output / "matched_validation_selection.csv", index=False)
    write_jsonl(output / "diagnostics_selected.jsonl", plan)
    print(f"Selected {len(plan)} checkpoints from validation only", flush=True)


def summarize_diagnostics(tasks: list[dict], output: Path, prefix: str) -> None:
    """Per-sample, per-water, overall and per-seed tables of completed tasks, named ``<prefix>_*.csv``."""
    frames, coverage = [], []
    for task in tasks:
        root = Path(task["destination"])
        if not (root / "completed.json").exists():
            raise ValueError(f"Unfinished diagnostic task: {root}")
        for split in ("train", "val", "test"):
            meta = json.loads((root / f"{split}.json").read_text())
            identity = {key: task[key] for key in KEYS}
            coverage.append(identity | meta)
            if meta["scored_sequences"] == 0:
                if split != "val":
                    raise ValueError(f"No scored {split} data: {root}")
                continue
            frame = pd.read_csv(root / f"{split}.csv")
            for key, value in identity.items():
                frame[key] = value
            frames.append(frame)
    per_sample = pd.concat(frames, ignore_index=True)
    # Both regimes must score the same scenes and pixels within a condition.
    scene_keys = ["data_amount", "seed", "water_id", "split", "sample_id"]
    if (per_sample.groupby(scene_keys).regime.nunique() != 2).any():
        raise ValueError("Unpaired model coverage in diagnostic scores")
    if (per_sample.groupby(scene_keys).n_valid_pixels.nunique() != 1).any():
        raise ValueError("Methods were evaluated on different reference pixels")
    per_water = per_sample.groupby(KEYS + ["split"])[METRICS].mean().reset_index()
    counts = per_sample.groupby(KEYS + ["split"]).size().rename("n_samples").reset_index()
    per_water = per_water.merge(counts, on=KEYS + ["split"], validate="one_to_one")
    overall = overall_means(per_water, ["regime", "data_amount", "seed", "split"], METRICS)
    seeds = seed_summary(overall, ["regime", "data_amount", "split"], METRICS)
    gaps = per_water.pivot(index=KEYS, columns="split", values=METRIC).reset_index()
    gaps["test_minus_train"] = gaps["test"] - gaps["train"]
    gaps["val_minus_train"] = gaps["val"] - gaps["train"]
    output.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(output / f"{prefix}_per_sample.csv", index=False)
    per_water.to_csv(output / f"{prefix}_per_water.csv", index=False)
    overall.to_csv(output / f"{prefix}_overall_seeds.csv", index=False)
    seeds.to_csv(output / f"{prefix}_summary.csv", index=False)
    gaps.to_csv(output / f"{prefix}_gaps_per_water.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / f"{prefix}_coverage.csv", index=False)
    print(f"Aggregated {len(tasks)} tasks, {len(per_sample)} scored scenes", flush=True)
