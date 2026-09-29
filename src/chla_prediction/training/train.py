"""Train a forecaster config: multi-water pretraining, few-shot transfer, or training from scratch on a target water."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from chla_prediction.config import OptimConfig, TrainForecasterConfig
from chla_prediction.imagery.dataset import SequenceDataset, SingleFrameDataset
from chla_prediction.imagery.waters import load_waters
from chla_prediction.models.registry import build_model
from chla_prediction.training.loop import run_training_phases

__all__ = ["TrainingOutput", "build_loaders", "git_hash", "make_loader", "train_forecaster"]


@dataclass(frozen=True)
class TrainingOutput:
    run_dir: Path
    manifest_path: Path
    sample_counts: dict[str, int]


def git_hash() -> str | None:
    """Commit of the running code; ``CHLA_GIT_HASH`` overrides it for copies without ``.git``."""
    if explicit := os.environ.get("CHLA_GIT_HASH"):
        return explicit
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def make_loader(
    dataset: Dataset, optim: OptimConfig, shuffle: bool, batch_size: int | None = None, prefetch: int = 4
) -> DataLoader:
    workers = optim.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size or optim.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
    )


def build_loaders(config: TrainForecasterConfig) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Loaders of the warm-up frames (``reconstruction``) and the training and validation sequences."""
    data = config.data
    archives, splits = load_waters(data)

    def sequences(split: str, random_crop: bool) -> SequenceDataset:
        samples = {water_id: water_splits[split] for water_id, water_splits in splits.items()}
        return SequenceDataset(archives, samples, data, random_crop, config.seed)

    # Warm-up frames come from the training period only.
    train_start = data.sequence.train_start or date.min
    train_period = [
        archive.with_scenes(tuple(s for s in archive.scenes if train_start <= s.scene_date <= data.sequence.train_end))
        for archive in archives
    ]
    frames = SingleFrameDataset(train_period, data, config.seed)
    train_sequences = sequences("train", random_crop=True)
    val_sequences = sequences("val", random_crop=False)
    if len(train_sequences) == 0:
        raise ValueError("No training sequences; check split boundaries against archive dates")
    loaders = {
        "reconstruction": make_loader(frames, config.optim, shuffle=True),
        "forecast": make_loader(train_sequences, config.optim, shuffle=True),
        "forecast_val": make_loader(val_sequences, config.optim, shuffle=False),
    }
    counts = {
        "reconstruction_frames": len(frames),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
    }
    return loaders, counts


def train_forecaster(config_path: Path) -> TrainingOutput:
    config = TrainForecasterConfig.from_yaml(config_path)
    torch.manual_seed(config.seed)
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, counts = build_loaders(config)
    model = build_model(config.model, config.data)
    if config.init_checkpoint is not None:
        model.load_state_dict(torch.load(config.init_checkpoint, map_location="cpu", weights_only=True))
    phase_results = run_training_phases(model, loaders, config, run_dir)

    manifest_path = run_dir / "pretrain_manifest.json"
    manifest = {
        "config": json.loads(config.model_dump_json()),
        "config_path": str(config_path),
        "git_hash": git_hash(),
        "seed": config.seed,
        "sample_counts": counts,
        "phases": [
            {
                "name": result.name,
                "epochs": result.epochs,
                "final_train_loss": result.final_train_loss,
                "final_val_loss": result.final_val_loss,
            }
            for result in phase_results
        ],
        "checkpoints": sorted(str(path) for path in run_dir.glob("checkpoint_*.pt")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TrainingOutput(run_dir=run_dir, manifest_path=manifest_path, sample_counts=counts)
