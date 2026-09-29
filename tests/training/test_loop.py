import torch

from chla_prediction.config import TrainForecasterConfig
from chla_prediction.training.loop import _forecast_step


class _ConstantForecaster(torch.nn.Module):
    """Forecasts 2 and reconstructs 0.5 everywhere."""

    def forecast(self, images, *args, **kwargs):
        return torch.full((images.shape[0], 1, 4, 4), 2.0)

    def reconstruct(self, image, valid):
        return torch.full((image.shape[0], 1, 4, 4), 0.5)


def test_forecast_step_adds_the_weighted_reconstruction_of_the_last_input(make_data_config) -> None:
    """Loss = masked L1 forecast loss + reconstruction_weight x reconstruction loss of the most recent scene."""
    batch = {
        "images": torch.rand(1, 3, 11, 4, 4),
        "masks": torch.ones(1, 3, 4, 4),
        "frame_valid": torch.ones(1, 3),
        "interval_days": torch.ones(1, 3),
        "lead_days": torch.ones(1),
        "input_targets": torch.zeros(1, 3, 1, 4, 4),
        "input_target_masks": torch.ones(1, 3, 4, 4),
        "target": torch.ones(1, 1, 4, 4),
        "target_valid": torch.ones(1, 4, 4),
    }
    config = TrainForecasterConfig(
        run_name="stub",
        data=make_data_config(target_product="mdn_chla"),
        objective={"reconstruction_weight": 0.3},
        phases=[{"name": "forecast", "objective": "forecast", "epochs": 1}],
    )
    loss = _forecast_step(_ConstantForecaster(), batch, config, torch.device("cpu"))
    assert torch.isclose(loss, torch.tensor(1.0 + 0.3 * 0.5))
