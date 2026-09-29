"""Training objectives shared by multi-water pretraining and few-shot transfer."""

from __future__ import annotations

import torch

from chla_prediction.config import ObjectiveConfig

__all__ = ["band_patch_mask", "masked_l1", "masked_reconstruction_loss", "next_scene_loss"]


def band_patch_mask(
    shape: tuple[int, ...],
    mask_ratio: float,
    patch_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean mask ``[B, C, H, W]`` hiding random square patches, drawn independently per sample and band."""
    batch, channels, height, width = shape
    grid_h = -(-height // patch_size)
    grid_w = -(-width // patch_size)
    coarse = torch.rand((batch, channels, grid_h, grid_w), generator=generator) < mask_ratio
    fine = coarse.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return fine[:, :, :height, :width]


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean absolute error weighted by ``weight``."""
    total = weight.sum().clamp_min(1.0)
    return ((prediction - target).abs() * weight).sum() / total


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    total = weight.sum().clamp_min(1.0)
    return ((prediction - target).square() * weight).sum() / total


def masked_reconstruction_loss(
    model,
    image: torch.Tensor,
    valid: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    config: ObjectiveConfig,
) -> torch.Tensor:
    """Reconstruction loss L_rc: corrupt random band patches of ``image`` and reconstruct its pseudo-label ``target``.

    Every valid target pixel is scored, so the corruption acts as input
    augmentation: the loss recovers the scene's pseudo-label from its
    corrupted reflectance.
    """
    hide = band_patch_mask(tuple(image.shape), config.mask_ratio, config.mask_patch_size)
    hide = hide.to(image.device)
    corrupted = image * (~hide).float()
    reconstruction = model.reconstruct(corrupted, valid)
    return masked_l1(reconstruction, target, target_valid.unsqueeze(1).expand_as(target))


def next_scene_loss(
    prediction: torch.Tensor, target: torch.Tensor, target_valid: torch.Tensor, config: ObjectiveConfig
) -> torch.Tensor:
    """Forecast loss L_fc: masked L1 (or MSE) against the target scene."""
    error = _masked_mse if config.prediction_loss == "mse" else masked_l1
    return error(prediction, target, target_valid.unsqueeze(1).expand_as(prediction))
