"""Convolutional decoder shared by the reconstruction and forecasting paths."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FieldDecoder"]


class FieldDecoder(nn.Module):
    """Upsample a latent map to input resolution and predict the log10 Chl-a field, unconstrained."""

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        base_channels: int,
        upsample_stages: int = 2,
    ):
        super().__init__()
        layers = []
        channels = latent_channels
        for stage in range(upsample_stages):
            width = base_channels * 2 ** (upsample_stages - stage - 1)
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels, width, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            ]
            channels = width
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=1))
        self.stages = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.stages(latent)
