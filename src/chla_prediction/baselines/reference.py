"""The Persistence and Climatology reference forecasts, which have no parameters."""

from __future__ import annotations

import torch

from chla_prediction.config import DataConfig, ModelConfig
from chla_prediction.models.forecaster import Forecaster, persistence_forecast

__all__ = ["ClimatologyForecaster", "PersistenceForecaster"]


class PersistenceForecaster(Forecaster):
    """Carry forward each pixel's most recent valid observation; NaN where none exists."""

    prediction_nodata = float("nan")

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()

    def forecast(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        frame_valid: torch.Tensor,
        interval_days: torch.Tensor,
        lead_days: torch.Tensor,
        input_targets: torch.Tensor,
        input_target_masks: torch.Tensor,
        **_: torch.Tensor | None,
    ) -> torch.Tensor:
        prediction = persistence_forecast(input_targets, input_target_masks, frame_valid)
        observed = (input_target_masks.bool() & frame_valid[:, :, None, None].bool()).any(dim=1)
        return prediction.masked_fill(~observed[:, None], float("nan"))


class ClimatologyForecaster(Forecaster):
    """The water's eight-day climatology of the target pseudo-label at the target date.

    ``base`` is NaN where the climatology is undefined, so those pixels are left out.
    """

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        if not data_config.climatology_base:
            raise ValueError("climatology_forecaster needs data.climatology_base")

    def forecast(self, *_: torch.Tensor, base: torch.Tensor, **__: torch.Tensor | None) -> torch.Tensor:
        return base
