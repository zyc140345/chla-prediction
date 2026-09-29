import torch

from chla_prediction.config import ObjectiveConfig
from chla_prediction.training.objectives import band_patch_mask, next_scene_loss


def test_band_patch_mask_ratio_and_shape() -> None:
    generator = torch.Generator().manual_seed(0)
    mask = band_patch_mask((4, 6, 32, 32), mask_ratio=0.5, patch_size=8, generator=generator)
    assert mask.shape == (4, 6, 32, 32)
    assert 0.3 < mask.float().mean().item() < 0.7


def test_next_scene_loss_ignores_invalid_pixels() -> None:
    config = ObjectiveConfig()
    prediction = torch.rand(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)
    assert next_scene_loss(prediction, target, torch.zeros(2, 8, 8), config).item() == 0.0
    assert next_scene_loss(prediction, target, torch.ones(2, 8, 8), config).item() > 0.0
