"""The forecaster interface and the irregular-interval-aware forecaster built from encoder, temporal module and decoder."""

from __future__ import annotations

import torch
from torch import nn

from chla_prediction.config import DataConfig, ModelConfig
from chla_prediction.models.clay import ClayFrozenEncoder
from chla_prediction.models.decoder import FieldDecoder
from chla_prediction.models.encoder import ConvNeXtEncoder, MDNSpectralEncoder, ResNetEncoder, SimpleCNNEncoder
from chla_prediction.models.temporal import (
    DAYS_SCALE,
)

__all__ = ["Forecaster", "IntervalAwareForecaster", "persistence_forecast"]


class Forecaster(nn.Module):
    """Base of every forecasting model, the proposed one and the baselines.

    ``forecast(images, masks, frame_valid, interval_days, lead_days,
    input_targets=, input_target_masks=, base=)`` returns the forecast field
    ``[B, C, H, W]`` in the output space.
    """

    # Inference pads scenes to a multiple of this.
    pad_multiple = 1
    # Nodata value of the written forecast rasters.
    prediction_nodata: float | None = None
    # Dropout stays active at inference, so each sample is forecast under its own seed.
    stochastic_inference = False


def persistence_forecast(frames: torch.Tensor, masks: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    """Most recent valid observation per pixel of ``frames`` ``[B, T, C, H, W]``; zero where none exists.

    ``masks`` ``[B, T, H, W]`` flags valid pixels and ``frame_valid`` ``[B, T]`` valid frames.
    """
    batch, steps, channels, height, width = frames.shape
    prediction = frames.new_zeros((batch, channels, height, width))
    filled = masks.new_zeros((batch, 1, height, width))
    for step in reversed(range(steps)):
        usable = masks[:, step : step + 1] * frame_valid[:, step].view(batch, 1, 1, 1)
        take = usable * (1.0 - filled)
        prediction = prediction + take.expand_as(prediction) * frames[:, step]
        filled = torch.clamp(filled + take, max=1.0)
    return prediction


class IntervalAwareForecaster(Forecaster):
    """Encode past frames, roll the temporal module, and decode the forecast field.

    ``recipe.PROPOSED_MODEL`` selects the ``mdn_spectral`` encoder and
    ``TargetRelativeAttention``. The decoder receives the forecast lead as an
    extra latent channel and predicts the log10 Chl-a pseudo-label field. The
    reconstruction path of the warm-up phase encodes one frame and decodes it
    with the same decoder.
    """

    def __init__(self, model_config: ModelConfig, data_config: DataConfig, temporal_cls: type):
        super().__init__()
        if model_config.encoder == "clay_frozen":
            self.encoder = ClayFrozenEncoder(
                band_names=data_config.model_input_bands(),
                checkpoint=model_config.clay_checkpoint,
                dim=model_config.clay_dim,
                depth=model_config.clay_depth,
                heads=model_config.clay_heads,
                lora_rank=model_config.clay_lora_rank,
            )
        elif model_config.encoder == "mdn_spectral":
            self.encoder = MDNSpectralEncoder(
                band_names=data_config.model_input_bands(),
                latent_channels=model_config.latent_channels,
                mdn_weights=model_config.mdn_weights,
            )
        elif model_config.encoder in ("resnet", "resnet50"):
            self.encoder = ResNetEncoder(
                in_channels=len(data_config.model_input_bands()),
                latent_channels=model_config.latent_channels,
                depth=50 if model_config.encoder == "resnet50" else 18,
            )
        elif model_config.encoder == "convnext":
            self.encoder = ConvNeXtEncoder(
                band_names=data_config.model_input_bands(),
                latent_channels=model_config.latent_channels,
                dim=model_config.encoder_width,
                depth=model_config.encoder_depth,
            )
        else:
            self.encoder = SimpleCNNEncoder(
                in_channels=len(data_config.model_input_bands()),
                base_channels=model_config.base_channels,
                latent_channels=model_config.latent_channels,
            )
        self.downsample_factor = self.encoder.downsample_factor
        temporal_options = {"depth": model_config.temporal_depth, "input_window": data_config.sequence.input_window}
        self.temporal = temporal_cls(
            model_config.hidden_channels,
            model_config.hidden_channels,
            **{key: value for key, value in temporal_options.items() if value is not None},
        )
        self.pad_multiple = self.downsample_factor * self.temporal.spatial_multiple
        self.time_conditioning = model_config.time_conditioning
        # Decoder input: hidden state and forecast lead.
        self.decoder = FieldDecoder(
            latent_channels=model_config.hidden_channels + 1,
            out_channels=len(data_config.model_output_bands()),
            base_channels=model_config.base_channels,
            upsample_stages=self.downsample_factor.bit_length() - 1,
        )
        self.latent_project = (
            nn.Identity()
            if self.encoder.latent_channels == model_config.hidden_channels
            else nn.Conv2d(self.encoder.latent_channels, model_config.hidden_channels, kernel_size=1)
        )

    def _encode(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.latent_project(self.encoder(frames, masks))

    def _decode(self, latent: torch.Tensor, lead_days: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = latent.shape
        lead = (lead_days / DAYS_SCALE).view(batch, 1, 1, 1).expand(batch, 1, height, width)
        return self.decoder(torch.cat([latent, lead], dim=1))

    def forecast(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        frame_valid: torch.Tensor,
        interval_days: torch.Tensor,
        lead_days: torch.Tensor,
        **_: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.time_conditioning:
            # Time-conditioning ablation: intervals and lead become constants.
            interval_days = torch.zeros_like(interval_days)
            lead_days = torch.zeros_like(lead_days)
        batch, steps, channels, height, width = images.shape
        factor = self.downsample_factor
        latents = self._encode(images.reshape(batch * steps, channels, height, width), masks.reshape(-1, height, width))
        latents = latents.reshape(batch, steps, -1, height // factor, width // factor)
        hidden = self.temporal(latents, interval_days, frame_valid, lead_days=lead_days)
        return self._decode(hidden, lead_days)

    def reconstruct(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._decode(self._encode(image, mask), image.new_zeros(image.shape[0]))
