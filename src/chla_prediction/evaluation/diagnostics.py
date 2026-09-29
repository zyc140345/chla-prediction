"""Score training checkpoints on full scenes of any split, as the evaluation scores test forecasts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch

from chla_prediction.config import EvaluateForecastConfig, TrainForecasterConfig
from chla_prediction.evaluation.scoring import SampleScorer, score_split, write_scores
from chla_prediction.imagery.climatology import water_climatology
from chla_prediction.imagery.masks import water_extent_masks
from chla_prediction.imagery.waters import load_waters
from chla_prediction.inference.predict import forecast_sample
from chla_prediction.models.registry import build_model
from chla_prediction.training.train import git_hash, train_forecaster

__all__ = ["METRIC", "CheckpointEvaluator", "run_task", "select_validation_checkpoint"]

METRIC = "rmse_log"


class CheckpointEvaluator:
    """Score checkpoints of one training config with next-observation targets and the evaluation's water masks."""

    def __init__(self, train: TrainForecasterConfig, evaluation: EvaluateForecastConfig):
        self.data = train.data.next_observation_only()
        self.evaluation = evaluation
        self.device = torch.device(train.optim.device)
        self.archives, self.splits = load_waters(self.data)
        self.water_masks = water_extent_masks(self.archives, evaluation)
        self.scorer = SampleScorer(evaluation)
        self.model = build_model(train.model, self.data).to(self.device).eval()
        self.climatologies = {archive.water_id: water_climatology(archive, self.data) for archive in self.archives}

    def score(self, checkpoint: Path, split: str, output: Path, epoch: int | None = None) -> pd.DataFrame:
        """Score one split; write per-scene metrics (CSV) and counts (JSON)."""
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))

        def predict(archive, sample):
            climatology = self.climatologies[archive.water_id]
            return forecast_sample(self.model, archive, sample, self.data, self.device, climatology)

        frame, candidates = score_split(
            predict,
            self.archives,
            self.splits,
            split,
            self.data.model_output_bands(),
            self.scorer,
            self.water_masks,
            self.evaluation.min_valid_fraction,
        )
        if len(frame):
            frame = frame.assign(split=split, epoch=epoch, checkpoint=str(checkpoint))
        write_scores(
            frame,
            output,
            checkpoint=str(checkpoint),
            split=split,
            epoch=epoch,
            candidate_sequences=candidates,
            scored_sequences=len(frame),
            task="next_observation",
            mean_rmse_log=float(frame[METRIC].mean()) if len(frame) else None,
        )
        return frame


def select_validation_checkpoint(candidates: pd.DataFrame) -> pd.Series:
    """Select by validation error alone; ties prefer the earlier epoch, then the lower learning rate."""
    validation = candidates[candidates["split"] == "val"].dropna(subset=[METRIC])
    if validation.empty:
        raise ValueError("Checkpoint selection requires nonempty validation scores")
    return validation.sort_values([METRIC, "epoch", "learning_rate"]).iloc[0]


def run_task(task: dict) -> None:
    """Run one plan task: score an existing checkpoint (``curve``, ``selected``) or train, then score (``matched``)."""
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    destination = Path(task["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    if task["kind"] == "curve":
        manifest = json.loads(Path(task["source_manifest"]).read_text())
        train = TrainForecasterConfig.model_validate(manifest["config"])
        checkpoint = Path(task["source_manifest"]).parent / "checkpoint_forecast.pt"
    elif task["kind"] == "selected":
        train = TrainForecasterConfig.from_yaml(task["train_config"])
        checkpoint = Path(task["checkpoint"])
    else:
        train = TrainForecasterConfig.from_yaml(task["train_config"])
        run_dir = train.output_dir / train.run_name
        # Skip training if the run finished; an interrupted run resumes from its own state.
        if not (run_dir / "pretrain_manifest.json").exists():
            train_forecaster(Path(task["train_config"]))
        checkpoint = run_dir / "checkpoint_forecast.pt"
    evaluation = EvaluateForecastConfig.from_yaml(task["evaluation_config"])
    evaluation = evaluation.model_copy(update={"data": train.data})
    evaluator = CheckpointEvaluator(train, evaluation)
    if task["kind"] in ("curve", "selected"):
        for split in ("train", "val", "test"):
            evaluator.score(checkpoint, split, destination / f"{split}.csv")
    else:
        trajectory = []
        for epoch in train.phases[0].checkpoint_epochs:
            checkpoint = run_dir / f"checkpoint_forecast_epoch{epoch:03d}.pt"
            for split in ("train", "val"):
                scored = evaluator.score(checkpoint, split, destination / f"epoch{epoch:03d}_{split}.csv", epoch)
                trajectory.append(
                    {
                        "epoch": epoch,
                        "split": split,
                        "learning_rate": train.optim.learning_rate,
                        METRIC: float(scored[METRIC].mean()),
                        "n_samples": len(scored),
                        "checkpoint": str(checkpoint),
                    }
                )
        pd.DataFrame(trajectory).to_csv(destination / "trajectory.csv", index=False)
    (destination / "completed.json").write_text(
        json.dumps(
            {
                "task": task,
                "source_revision": git_hash(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            indent=2,
        )
    )
