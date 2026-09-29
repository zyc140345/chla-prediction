"""Per-frame encoders mapping a scene and its validity mask to a latent channel map at 1/4 resolution.

The proposed spectral-spatial encoder (``MDNSpectralEncoder``, spectral stage
initialized from the pretrained MDN), the small CNN trained from scratch, and
the torchvision ResNet / ConvNeXt trunks of the encoder ablation. Clay lives
in models/clay.py and runs at 1/8 resolution.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from chla_prediction.retrieval.mdn import MDN_BANDS, read_mdn_export, surface_reflectance_to_rrs

__all__ = ["ConvNeXtEncoder", "ResNetEncoder", "SimpleCNNEncoder", "MDNSpectralEncoder"]


def _block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.SiLU(inplace=True),
    )


def _masked(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Zero invalid pixels and append the validity mask, so later layers tell masked pixels from small values."""
    mask_channel = masks.unsqueeze(1)
    return torch.cat([features * mask_channel, mask_channel], dim=1)


class SimpleCNNEncoder(nn.Module):
    """Four-block CNN (two strided blocks) producing a latent map at 1/4 input resolution."""

    def __init__(self, in_channels: int, base_channels: int, latent_channels: int):
        super().__init__()
        self.stages = nn.Sequential(
            _block(in_channels + 1, base_channels, stride=1),
            _block(base_channels, base_channels * 2, stride=2),
            _block(base_channels * 2, latent_channels, stride=2),
            _block(latent_channels, latent_channels, stride=1),
        )
        self.latent_channels = latent_channels
        self.downsample_factor = 4

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.stages(_masked(frames, masks))


class MDNSpectralEncoder(nn.Module):
    """Spectral-spatial encoder whose spectral stage mirrors the MDN's hidden stack.

    The spectral stage runs 1x1 convolutions with ReLU over the seven MDN
    bands, initialized from the first member of the exported MDN ensemble
    when ``mdn_weights`` is given; two strided blocks then reach the
    1/4-resolution latent grid.
    """

    def __init__(self, band_names: list[str], latent_channels: int, mdn_weights: Path | None):
        super().__init__()
        self.band_indices = [band_names.index(band) for band in MDN_BANDS]
        layers: list[nn.Module] = []
        widths = [len(MDN_BANDS)] + [100] * 5
        for i in range(5):
            layers += [nn.Conv2d(widths[i], widths[i + 1], kernel_size=1), nn.ReLU(inplace=True)]
        self.spectral = nn.Sequential(*layers)
        self.register_buffer("x_center", torch.zeros(len(MDN_BANDS)))
        self.register_buffer("x_scale", torch.ones(len(MDN_BANDS)))
        if mdn_weights is not None:
            member = read_mdn_export(mdn_weights)[0]
            convs = [module for module in self.spectral if isinstance(module, nn.Conv2d)]
            with torch.no_grad():
                self.x_center.copy_(torch.as_tensor(member["x_center"], dtype=torch.float32))
                self.x_scale.copy_(torch.as_tensor(member["x_scale"], dtype=torch.float32))
                for conv, kernel, bias in zip(convs, member["weights"][0::2], member["weights"][1::2], strict=False):
                    conv.weight.copy_(torch.from_numpy(kernel.T.copy())[:, :, None, None])
                    conv.bias.copy_(torch.from_numpy(bias.copy()))
        self.spatial = nn.Sequential(
            _block(100 + 1, latent_channels, stride=2),
            _block(latent_channels, latent_channels, stride=2),
        )
        self.latent_channels = latent_channels
        self.downsample_factor = 4

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # The MDN's input convention: Rrs = reflectance / pi, then its robust scaling.
        rrs = surface_reflectance_to_rrs(frames[:, self.band_indices])
        scaled = (rrs - self.x_center.view(1, -1, 1, 1)) / self.x_scale.view(1, -1, 1, 1)
        features = self.spectral(scaled * masks.unsqueeze(1))
        return self.spatial(_masked(features, masks))


class ResNetEncoder(nn.Module):
    """torchvision ResNet-18/50 truncated after layer3 at output stride 4, then a 1x1 projection.

    The stride-2 entries of layers 2 and 3 are set to 1, so the 256 (1024)
    output channels stay at 1/4 resolution. GroupNorm replaces BatchNorm
    because batches hold 4-8 sequences.
    """

    def __init__(self, in_channels: int, latent_channels: int, depth: int = 18):
        super().__init__()
        from torchvision.models import resnet18, resnet50

        trunk = {18: resnet18, 50: resnet50}[depth](weights=None, norm_layer=_group_norm)
        trunk.conv1 = nn.Conv2d(in_channels + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for layer in (trunk.layer2, trunk.layer3):
            # Bottleneck strides on conv2, BasicBlock on conv1.
            strided = layer[0].conv2 if hasattr(layer[0], "conv3") else layer[0].conv1
            strided.stride = (1, 1)
            layer[0].downsample[0].stride = (1, 1)
        self.trunk = nn.Sequential(
            trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool, trunk.layer1, trunk.layer2, trunk.layer3
        )
        self.project = nn.Conv2d(256 if depth == 18 else 1024, latent_channels, kernel_size=1)
        self.latent_channels = latent_channels
        self.downsample_factor = 4

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.project(self.trunk(_masked(frames, masks)))


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=8, num_channels=channels)


class ConvNeXtEncoder(nn.Module):
    """One torchvision ConvNeXt stage at 1/4 resolution: 4x4 stride-4 stem, ``depth`` blocks, 1x1 projection."""

    def __init__(
        self,
        band_names: list[str],
        latent_channels: int,
        dim: int = 128,
        depth: int = 6,
    ):
        super().__init__()
        from torchvision.models.convnext import CNBlockConfig, ConvNeXt

        in_channels = len(band_names) + 1
        model = ConvNeXt([CNBlockConfig(input_channels=dim, out_channels=None, num_layers=depth)])
        model.features[0][0] = nn.Conv2d(in_channels, dim, kernel_size=4, stride=4)
        self.body = model.features
        self.project = nn.Conv2d(dim, latent_channels, kernel_size=1)
        self.latent_channels = latent_channels
        self.downsample_factor = 4

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = _masked(frames, masks)
        return self.project(self.body(x))
