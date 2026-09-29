"""Yao et al. (2023) stacked ConvLSTM, the deep baseline trained from scratch on each target water.

Yao et al. (2023), doi:10.3390/rs15184486, forecast monthly Chl-a fields
with four stacked standard ConvLSTM layers (their equations 3-8: gates over
``[H, X]`` only, no peephole terms) at full resolution, trained with MSE and
Adam under ReduceLROnPlateau. The stack runs directly on the log10 MDN
pseudo-labels and a 1x1 convolution reads out the forecast field.

Adaptations to the data:

- The paper's inputs are gap-free composites; here invalid pixels take
  their frame's valid mean, and padded frames leave the recurrent state
  untouched.
- The paper min-max normalizes per variable; the fixed log10 Chl-a range
  plays that role, so no statistic is fitted.
- The paper feeds three monthly frames; here every variant gets the same
  six-scene samples, so every variant is scored on the same samples.
- The paper does not report the hidden width; this reproduction uses 64
  channels per layer.
"""

from __future__ import annotations

import torch
from torch import nn

from chla_prediction.baselines.frames import fill_gaps, from_unit, to_unit
from chla_prediction.config import DataConfig, ModelConfig
from chla_prediction.models.forecaster import Forecaster
from chla_prediction.models.temporal import ConvLSTMCell

__all__ = ["StackedConvLSTMForecaster"]


class StackedConvLSTMForecaster(Forecaster):
    """Standard ConvLSTM layers rolled over the pseudo-label frames; the last hidden state is read out."""

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        if data_config.target_product != "mdn_chla":
            raise ValueError(
                "stacked_conv_lstm_forecaster consumes the pseudo-label channel; set data.target_product: mdn_chla"
            )
        hidden = model_config.hidden_channels
        depth = model_config.temporal_depth or 4
        self.cells = nn.ModuleList(ConvLSTMCell(1 if index == 0 else hidden, hidden) for index in range(depth))
        self.readout = nn.Conv2d(hidden, 1, kernel_size=1)

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
        del images, masks, interval_days, lead_days
        frames = to_unit(fill_gaps(input_targets[:, :, 0], input_target_masks))
        batch, steps, height, width = frames.shape
        hidden = [frames.new_zeros((batch, cell.hidden_channels, height, width)) for cell in self.cells]
        cell_state = [state.clone() for state in hidden]
        for step in range(steps):
            layer_input = frames[:, step : step + 1]
            keep = 1.0 - frame_valid[:, step].view(batch, 1, 1, 1)
            for index, cell in enumerate(self.cells):
                new_hidden, new_cell = cell(layer_input, (hidden[index], cell_state[index]))
                hidden[index] = frame_valid[:, step].view(batch, 1, 1, 1) * new_hidden + keep * hidden[index]
                cell_state[index] = frame_valid[:, step].view(batch, 1, 1, 1) * new_cell + keep * cell_state[index]
                layer_input = hidden[index]
        return from_unit(self.readout(hidden[-1]))
