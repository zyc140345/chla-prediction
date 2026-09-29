"""Typed configuration for the Chl-a field forecasting experiments."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chla_prediction.recipe import (
    CHLA_RANGE,
    CROP_SIZE,
    FORECAST_OBJECTIVE,
    MAX_SCENE_CLOUD_FRACTION,
    PRETRAINING_OPTIM,
    PROPOSED_MODEL,
    SEQUENCE,
    SHORELINE_EROSION_PX,
    WATER_MASK_BAND,
    WATER_MASK_THRESHOLD,
)

ALL_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# The MDN's 443-783 nm bands.
DEFAULT_OUTPUT_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
PRODUCT_CHANNELS = {
    "mdn_chla": "LOG10_CHLA",
    "ndci_chla": "LOG10_CHLA_NDCI",
    "three_band_chla": "LOG10_CHLA_3BDA",
}

Product = Literal["mdn_chla", "ndci_chla", "three_band_chla"]
Split = Literal["train", "val", "test"]


def pseudo_label_of(bands: list[str]) -> str | None:
    """Pseudo-label forecast by a model with these output bands; None for reflectance."""
    return next((name for name, channel in PRODUCT_CHANNELS.items() if bands == [channel]), None)


__all__ = [
    "ALL_BANDS",
    "PRODUCT_CHANNELS",
    "DataConfig",
    "EvaluateForecastConfig",
    "InferenceForecastConfig",
    "ModelConfig",
    "ObjectiveConfig",
    "PhaseConfig",
    "PrecomputeClimatologyConfig",
    "PrecomputePseudoLabelsConfig",
    "SequenceConfig",
    "TabularBaselineConfig",
    "TrainForecasterConfig",
    "pseudo_label_of",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class _WaterMaskFields(_StrictModel):
    """Water-extent-mask keys shared by the evaluation and the tabular baseline, so both score the same pixels."""

    # External water-body band thresholded into the water-extent mask (imagery/masks.py).
    water_mask_dir: Path = Path("data/water_masks")
    water_mask_band: Literal["s1_median_water"] = WATER_MASK_BAND
    water_mask_threshold: float = Field(default=WATER_MASK_THRESHOLD, ge=0.0, le=100.0)
    # Rings of shoreline pixels dropped from the mask.
    shoreline_erosion_px: int = Field(default=SHORELINE_EROSION_PX, ge=0, le=5)


class ArchiveConfig(_StrictModel):
    water_id: str
    # STAC download manifest; its scene paths are relative to its directory.
    manifest_path: Path


class SequenceConfig(_StrictModel):
    input_window: int = Field(default=SEQUENCE["input_window"], ge=2)
    min_input_observations: int = Field(default=SEQUENCE["min_input_observations"], ge=1)
    train_end: date
    val_end: date
    # Training samples start on or after this date (data-amount study);
    # validation and test are unchanged.
    train_start: date | None = None
    # Also train on every later scene within this many days of the origin;
    # validation and test keep next-observation targets.
    train_max_lead_days: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_windows(self) -> SequenceConfig:
        if self.min_input_observations > self.input_window:
            raise ValueError("min_input_observations must not exceed input_window")
        if self.val_end <= self.train_end:
            raise ValueError("val_end must be after train_end")
        if self.train_start is not None and self.train_start >= self.train_end:
            raise ValueError("train_start must be before train_end")
        return self


class DataConfig(_StrictModel):
    archives: list[ArchiveConfig] = Field(default_factory=list)
    archive_root: Path | None = None
    sequence: SequenceConfig
    input_bands: list[str] = Field(default_factory=lambda: list(ALL_BANDS))
    output_bands: list[str] = Field(default_factory=lambda: list(DEFAULT_OUTPUT_BANDS))
    max_scene_cloud_fraction: float = Field(default=MAX_SCENE_CLOUD_FRACTION, ge=0.0, le=1.0)
    crop_size: int = Field(default=CROP_SIZE, ge=32)
    spatial_sampling: Literal["crop", "resize"] = "crop"
    exclude_waters: list[str] = Field(default_factory=list)
    # Keep a seeded random sample of this many waters (corpus-size study);
    # water ids carry geography, so the first N by name would not do.
    max_archives: int | None = Field(default=None, ge=1)
    # Use this log10 Chl-a pseudo-label as input and target instead of
    # reflectance (the per-pixel regressors and Climatology). ``mdn_chla`` is
    # read from the precomputed raster; ``ndci_chla`` and ``three_band_chla``
    # are computed from reflectance.
    input_product: Product | None = None
    # Reflectance input with this log10 Chl-a pseudo-label as the target (every trained model).
    target_product: Product | None = None
    # Load the eight-day climatology of the target pseudo-label
    # (precompute_climatology) as ``base``; needed by ``climatology_forecaster``.
    climatology_base: bool = False

    @model_validator(mode="after")
    def validate_bands(self) -> DataConfig:
        if not self.archives and self.archive_root is None:
            raise ValueError("Provide archives or archive_root")
        unknown = [band for band in self.input_bands + self.output_bands if band not in ALL_BANDS]
        if unknown:
            raise ValueError(f"Unknown bands: {unknown}; archive bands are {ALL_BANDS}")
        missing = [band for band in self.output_bands if band not in self.input_bands]
        if missing:
            raise ValueError(f"output_bands must be a subset of input_bands, missing {missing}")
        if self.input_product and self.target_product:
            raise ValueError(
                "input_product (pseudo-label inputs) and target_product (pseudo-label target) are exclusive"
            )
        if self.climatology_base and not self.forecast_product:
            raise ValueError("climatology_base needs a pseudo-label target (input_product or target_product)")
        return self

    @property
    def forecast_product(self) -> str | None:
        """Pseudo-label the model forecasts, or None for reflectance."""
        return self.input_product or self.target_product

    def archive_entries(self) -> list[ArchiveConfig]:
        """Explicit archives plus one per ``archive_root`` subdirectory with a STAC manifest, named after it."""
        entries = list(self.archives)
        if self.archive_root is not None:
            manifests = sorted(self.archive_root.glob("*/stac_download_manifest.jsonl"))
            entries += [ArchiveConfig(water_id=path.parent.name, manifest_path=path) for path in manifests]
        entries = [entry for entry in entries if entry.water_id not in self.exclude_waters]
        if not entries:
            raise ValueError(f"No archives found under {self.archive_root}")
        if self.max_archives is not None and self.max_archives < len(entries):
            index = np.random.default_rng(0).permutation(len(entries))
            keep = sorted(int(i) for i in index[: self.max_archives])
            entries = [entries[i] for i in keep]
        return entries

    def next_observation_only(self) -> DataConfig:
        """This config with next-observation training targets only."""
        return self.model_copy(update={"sequence": self.sequence.model_copy(update={"train_max_lead_days": None})})

    def model_input_bands(self) -> list[str]:
        return [PRODUCT_CHANNELS[self.input_product]] if self.input_product else self.input_bands

    def model_output_bands(self) -> list[str]:
        return [PRODUCT_CHANNELS[self.forecast_product]] if self.forecast_product else self.output_bands


class ModelConfig(_StrictModel):
    """Model architecture; the proposed model is ``recipe.PROPOSED_MODEL``."""

    # Registered model name (models/registry.py).
    name: str = "conv_gru_forecaster"
    base_channels: int = Field(default=PROPOSED_MODEL["base_channels"], ge=4)
    latent_channels: int = Field(default=PROPOSED_MODEL["latent_channels"], ge=8)
    hidden_channels: int = Field(default=PROPOSED_MODEL["hidden_channels"], ge=8)
    # Number of temporal-module layers; None keeps each module's default.
    temporal_depth: int | None = Field(default=None, ge=1)
    # False zeroes the acquisition intervals and the forecast lead at the
    # model input (time-conditioning ablation).
    time_conditioning: bool = True
    # Per-frame encoder (models/encoder.py, models/clay.py).
    encoder: Literal["simple_cnn", "clay_frozen", "mdn_spectral", "resnet", "resnet50", "convnext"] = "simple_cnn"
    # ConvNeXt stage width and depth.
    encoder_width: int = Field(default=128, ge=8)
    encoder_depth: int = Field(default=6, ge=1)
    clay_checkpoint: Path | None = None
    clay_dim: int = Field(default=1024, ge=8)
    clay_depth: int = Field(default=24, ge=1)
    clay_heads: int = Field(default=16, ge=1)
    # LoRA rank on the Clay attention projections; 0 freezes the encoder.
    clay_lora_rank: int = Field(default=8, ge=0)
    # MDN weights that initialize the ``mdn_spectral`` spectral stage; None
    # keeps a random initialization.
    mdn_weights: Path | None = None


class ObjectiveConfig(_StrictModel):
    """Training losses; the proposed model uses ``recipe.FORECAST_OBJECTIVE``."""

    mask_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    mask_patch_size: int = Field(default=8, ge=1)
    # Forecast loss norm; ``mse`` is for the stacked ConvLSTM baseline.
    prediction_loss: Literal["l1", "mse"] = "l1"
    reconstruction_weight: float = Field(default=FORECAST_OBJECTIVE["reconstruction_weight"], ge=0.0)


class PhaseConfig(_StrictModel):
    name: str
    # ``forecast`` adds the weighted reconstruction loss to the forecast loss.
    objective: Literal["reconstruction", "forecast"]
    # Epoch cap; with ``patience``, early stopping on the validation loss
    # restores the best weights.
    epochs: int = Field(ge=1)
    patience: int | None = Field(default=None, ge=1)
    # Save weights after these completed epochs; 0 saves the starting weights.
    checkpoint_epochs: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_checkpoint_epochs(self) -> PhaseConfig:
        if any(epoch < 0 or epoch > self.epochs for epoch in self.checkpoint_epochs):
            raise ValueError("checkpoint_epochs must lie between zero and the phase epoch budget")
        return self


class OptimConfig(_StrictModel):
    learning_rate: float = Field(default=PRETRAINING_OPTIM["learning_rate"], gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=PRETRAINING_OPTIM["batch_size"], ge=1)
    num_workers: int = Field(default=4, ge=0)
    grad_clip_norm: float = Field(default=1.0, gt=0.0)
    device: str = "cuda"
    # bfloat16 autocast on CUDA (no gradient scaler needed).
    amp: bool = True
    # ReduceLROnPlateau patience on the validation loss; None keeps the rate fixed.
    lr_plateau_patience: int | None = Field(default=None, ge=1)
    # Steps between the rolling-mean records of training_steps.jsonl.
    step_log_interval: int = Field(default=100, ge=1)


class TrainForecasterConfig(_StrictModel):
    run_name: str
    seed: int = Field(default=42, ge=0)
    output_dir: Path = Path("outputs/runs")
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    phases: list[PhaseConfig] = Field(min_length=1)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    init_checkpoint: Path | None = None

    @model_validator(mode="after")
    def validate_target(self) -> TrainForecasterConfig:
        # Every trained model forecasts a pseudo-label from reflectance.
        if self.data.target_product is None:
            raise ValueError("Training needs a pseudo-label target; set data.target_product")
        return self


class InferenceForecastConfig(_StrictModel):
    seed: int = Field(default=42, ge=0)
    run_name: str
    output_dir: Path = Path("outputs/runs")
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    checkpoint: Path | None = None
    split: Split = "test"
    device: str = "cuda"


class EvaluateForecastConfig(_WaterMaskFields):
    run_name: str
    output_dir: Path = Path("outputs/runs")
    data: DataConfig
    # Forecasts to score, by name; checkpoint diagnostics score in memory instead.
    prediction_manifests: dict[str, Path] = Field(default_factory=dict)
    skill_reference: str = "persistence"
    # Minimum share of the water-extent mask with a usable observation for a
    # date to be scored.
    min_valid_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    # Retrieval applied to predicted and observed reflectance for the Chl-a
    # metrics; pseudo-label forecasts enter them directly.
    chla_axis: Literal["mdn", "ndci", "three_band"] = "mdn"
    mdn_weights: Path | None = None
    chla_range: tuple[float, float] = CHLA_RANGE
    # Upper edges (days) of the forecast-lead bins: [7, 15, 30] gives <=7d,
    # 8-15d, 16-30d and >30d.
    horizon_bins: list[int] = Field(default_factory=lambda: [7, 15, 30])

    @model_validator(mode="after")
    def validate_axis(self) -> EvaluateForecastConfig:
        if self.chla_axis == "mdn" and self.mdn_weights is None:
            raise ValueError("chla_axis mdn needs mdn_weights")
        return self


class TabularBaselineConfig(_WaterMaskFields):
    """Per-pixel random forest and XGBoost baselines (baselines/tabular.py).

    One model per water, fitted on its adaptation set of the ``data.input_product`` pseudo-label.
    """

    run_name: str
    output_dir: Path = Path("outputs/runs")
    data: DataConfig
    split: Split = "test"
    seed: int = Field(default=42, ge=0)
    # Pixels drawn per training sample, and the row budget per water.
    pixels_per_sample: int = Field(default=256, ge=1)
    max_train_rows: int = Field(default=400_000, ge=1000)
    estimator: Literal["xgboost", "random_forest"] = "xgboost"
    device: Literal["cpu", "cuda"] = "cpu"
    max_iter: int = Field(default=300, ge=1)
    learning_rate: float = Field(default=0.06, gt=0.0)
    # None leaves the depth unbounded (XGBoost keeps its 31-leaf cap).
    max_depth: int | None = Field(default=None, ge=1)
    n_jobs: int = Field(default=-1)

    @model_validator(mode="after")
    def validate_product(self) -> TabularBaselineConfig:
        # The lag features are read from the input frames.
        if self.data.input_product is None:
            raise ValueError("The tabular baseline reads and forecasts a retrieval product; set data.input_product")
        return self


class PrecomputePseudoLabelsConfig(_StrictModel):
    """Attach the MDN log10 Chl-a raster (``mdn_chla``) to every scene of the configured archives."""

    data: DataConfig
    mdn_weights: Path
    chla_range: tuple[float, float] = CHLA_RANGE
    device: str = "cuda"


class PrecomputeClimatologyConfig(_StrictModel):
    """Write each water's eight-day climatology of a pseudo-label, fitted up to ``sequence.train_end``."""

    data: DataConfig
    product: Literal["mdn_chla"] = "mdn_chla"
