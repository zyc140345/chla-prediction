"""Nagkoulis et al. (2024) pix2pix, reproduced as the zero-shot transfer baseline.

Nagkoulis et al. (2024), doi:10.1029/2024WR037138, train a pix2pix
conditional GAN on 15 European lakes and apply it frozen to 3 unseen lakes.
It is reproduced in its own terms: the standard U-Net generator and 70x70
PatchGAN discriminator, the paper's three-frame packing, and adversarial
plus L1 training.

Input channels carry Chl-a maps at ``(t-2, t-1, t)`` and output channels
``(t-1, t, t+1)``. Only ``t+1`` is a forecast, so only it is scored.

The data adaptation uses the log10 MDN pseudo-labels, chronological
samples, gap filling and validation early stopping. Scenes are resized whole
to the generator's 256-pixel grid with nearest-neighbor sampling, as in the
source. The numerical details follow the TensorFlow pix2pix tutorial
(Apache-2.0) that the authors' code uses: BCE, bottleneck BN, dropout active
at inference, and current-input BN statistics.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from chla_prediction.baselines.frames import fill_gaps, from_unit, to_unit
from chla_prediction.config import DataConfig, ModelConfig
from chla_prediction.models.forecaster import Forecaster

__all__ = [
    "FRAMES",
    "PatchDiscriminator",
    "Pix2PixForecaster",
    "UNetGenerator",
    "from_tanh",
    "pack_frames",
    "to_tanh",
]

# Frames per packing, input and output.
FRAMES = 3


def to_tanh(values: torch.Tensor) -> torch.Tensor:
    """Log10 Chl-a onto the generator's tanh range [-1, 1]."""
    return to_unit(values) * 2.0 - 1.0


def from_tanh(values: torch.Tensor) -> torch.Tensor:
    return from_unit((values + 1.0) / 2.0)


class CurrentBatchNorm(nn.BatchNorm2d):
    """BatchNorm on current-batch statistics (TF ``training=True``); one value per channel returns the bias."""

    def __init__(self, channels: int):
        super().__init__(channels, eps=1e-3, momentum=0.01, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() // x.shape[1] == 1:
            return (x - x) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return super().forward(x)


class ActiveDropout(nn.Dropout):
    """pix2pix applies dropout at training, validation, and inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(CurrentBatchNorm(out_channels))
        layers.append(nn.LeakyReLU(0.3, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            CurrentBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(ActiveDropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """The pix2pix U-Net: eight stride-2 blocks and their mirror, dropout on the first three decoder blocks."""

    def __init__(self, in_channels: int = FRAMES, out_channels: int = FRAMES, width: int = 64):
        super().__init__()
        self.down1 = _Down(in_channels, width, normalize=False)
        self.down2 = _Down(width, width * 2)
        self.down3 = _Down(width * 2, width * 4)
        self.down4 = _Down(width * 4, width * 8)
        self.down5 = _Down(width * 8, width * 8)
        self.down6 = _Down(width * 8, width * 8)
        self.down7 = _Down(width * 8, width * 8)
        self.down8 = _Down(width * 8, width * 8)
        self.up1 = _Up(width * 8, width * 8, dropout=True)
        self.up2 = _Up(width * 16, width * 8, dropout=True)
        self.up3 = _Up(width * 16, width * 8, dropout=True)
        self.up4 = _Up(width * 16, width * 8)
        self.up5 = _Up(width * 16, width * 4)
        self.up6 = _Up(width * 8, width * 2)
        self.up7 = _Up(width * 4, width)
        self.final = nn.Sequential(nn.ConvTranspose2d(width * 2, out_channels, 4, 2, 1), nn.Tanh())
        self.apply(_initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u = self.up1(d8, d7)
        u = self.up2(u, d6)
        u = self.up3(u, d5)
        u = self.up4(u, d4)
        u = self.up5(u, d3)
        u = self.up6(u, d2)
        u = self.up7(u, d1)
        return self.final(u)


class PatchDiscriminator(nn.Module):
    """The 70x70 PatchGAN, judging the input and output packings together."""

    def __init__(self, in_channels: int = 2 * FRAMES, width: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, width, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
            CurrentBatchNorm(width * 2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False),
            CurrentBatchNorm(width * 4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(width * 4, width * 8, 4, 1, 1, bias=False),
            CurrentBatchNorm(width * 8),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(width * 8, 1, 4, 1, 1),
        )

        self.apply(_initialize)

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([condition, image], dim=1))


def pack_frames(input_targets: torch.Tensor, input_target_masks: torch.Tensor) -> torch.Tensor:
    """The three most recent Chl-a frames as one conditioning image."""
    return to_tanh(fill_gaps(input_targets[:, -FRAMES:, 0], input_target_masks[:, -FRAMES:]))


class Pix2PixForecaster(Forecaster):
    """The generator as a forecaster; the discriminator lives only in training (training/pix2pix.py)."""

    stochastic_inference = True

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__()
        del model_config
        if data_config.target_product != "mdn_chla":
            raise ValueError(
                "pix2pix_forecaster packs a single pseudo-label channel; set data.target_product: mdn_chla"
            )
        if data_config.sequence.input_window < FRAMES:
            raise ValueError(f"pix2pix_forecaster needs at least {FRAMES} input frames")
        if data_config.spatial_sampling != "resize":
            raise ValueError("pix2pix_forecaster works on whole resized scenes; set data.spatial_sampling: resize")
        self.generator = UNetGenerator()
        self.resize_size = data_config.crop_size

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
        del images, masks, frame_valid, interval_days, lead_days
        native_size = input_targets.shape[-2:]
        shape = input_targets.shape
        input_targets = F.interpolate(
            input_targets.reshape(shape[0], -1, *native_size),
            (self.resize_size, self.resize_size),
            mode="nearest-exact",
        ).reshape(*shape[:3], self.resize_size, self.resize_size)
        input_target_masks = F.interpolate(
            input_target_masks.float(), (self.resize_size, self.resize_size), mode="nearest-exact"
        )
        # Gaps are filled after resizing, as in the training dataset.
        packed = pack_frames(input_targets, input_target_masks)
        prediction = from_tanh(self.generator(packed)[:, FRAMES - 1 : FRAMES])
        return F.interpolate(prediction, native_size, mode="nearest-exact")
