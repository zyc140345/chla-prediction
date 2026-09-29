"""Full-scene forecasting inference writing rasters and a prediction manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from chla_prediction.config import DataConfig, InferenceForecastConfig
from chla_prediction.imagery.archive import SceneArchive
from chla_prediction.imagery.climatology import Climatology, water_climatology
from chla_prediction.imagery.dataset import load_input_frames
from chla_prediction.imagery.sequences import SequenceSample
from chla_prediction.imagery.waters import load_waters
from chla_prediction.inference.manifest import PredictionOutput, prediction_row, sample_id, write_prediction_raster
from chla_prediction.io import write_jsonl
from chla_prediction.models.registry import build_model

__all__ = ["forecast_sample", "load_model", "run_inference"]


def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    return functional.pad(tensor, (0, (-width) % multiple, 0, (-height) % multiple))


def load_model(config: InferenceForecastConfig) -> nn.Module:
    """The configured model with its checkpoint, on the configured device, in eval mode."""
    model = build_model(config.model, config.data)
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint, map_location="cpu", weights_only=True))
    return model.to(config.device).eval()


@torch.no_grad()
def forecast_sample(
    model: nn.Module,
    archive: SceneArchive,
    sample: SequenceSample,
    data: DataConfig,
    device: torch.device,
    climatology: Climatology | None,
) -> np.ndarray:
    """Forecast one sample on the full scene; ``[C, H, W]`` float32."""
    frames, valids, bases, base_valids = load_input_frames(archive, sample, data)
    height, width = frames[0].shape[-2:]

    def batch(arrays: list[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(arrays)).unsqueeze(0).to(device)
        return _pad_to_multiple(tensor.float(), model.pad_multiple)

    # Only the Climatology forecast reads ``base``.
    extra = {}
    if climatology is not None:
        base = climatology.base(archive.scenes[sample.target_index].scene_date).astype(np.float32)
        extra["base"] = _pad_to_multiple(torch.from_numpy(base)[None, None].to(device), model.pad_multiple)
    prediction = model.forecast(
        batch(frames),
        batch(valids),
        torch.ones((1, len(frames)), device=device),
        torch.tensor([list(sample.interval_days)], dtype=torch.float32, device=device),
        torch.tensor([float(sample.lead_days)], device=device),
        input_targets=batch(bases),
        input_target_masks=batch(base_valids),
        **extra,
    )
    return prediction[0, :, :height, :width].float().cpu().numpy()


def _seeded_forecast(key: str, device: torch.device, forecast: Callable[[], np.ndarray]) -> tuple[np.ndarray, int]:
    """Run ``forecast`` under a seed derived from ``key``, with deterministic convolutions."""
    sample_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4])
    devices = [device.index or 0] if device.type == "cuda" else []
    with (
        torch.random.fork_rng(devices=devices),
        torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False),
    ):
        torch.manual_seed(sample_seed)
        return forecast(), sample_seed


def run_inference(config_path: Path) -> PredictionOutput:
    config = InferenceForecastConfig.from_yaml(config_path)
    data = config.data
    device = torch.device(config.device)
    run_dir = config.output_dir / config.run_name
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config)
    archives, splits = load_waters(data)
    output_bands = data.model_output_bands()
    rows = []
    for archive in archives:
        climatology = water_climatology(archive, data)
        for sample in splits[archive.water_id][config.split]:
            forecast = partial(forecast_sample, model, archive, sample, data, device, climatology)
            fields = {}
            if model.stochastic_inference:
                prediction, seed = _seeded_forecast(f"{config.seed}:{sample_id(sample)}", device, forecast)
                fields = {
                    "inference_seed": seed,
                    "inference_rule": "current_image_bn_dropout",
                    "deterministic_cudnn": True,
                }
            else:
                prediction = forecast()
            target_scene = archive.scenes[sample.target_index]
            prediction_path = prediction_dir / f"{sample_id(sample)}.tif"
            write_prediction_raster(prediction_path, prediction, target_scene, output_bands, model.prediction_nodata)
            rows.append(
                prediction_row(
                    sample,
                    target_scene,
                    prediction_path,
                    output_bands,
                    split=config.split,
                    model_name=config.model.name,
                    checkpoint=str(config.checkpoint) if config.checkpoint else None,
                    **fields,
                )
            )
    manifest_path = run_dir / "prediction_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    return PredictionOutput(run_dir=run_dir, manifest_path=manifest_path, prediction_count=len(rows))
