import pytest
import torch

from chla_prediction.models.forecaster import persistence_forecast
from chla_prediction.models.registry import build_model


@pytest.mark.parametrize("name", ["conv_gru_forecaster", "conv_lstm_forecaster"])
def test_forecast_and_reconstruction_shapes(name, make_model_config, make_data_config, make_forecast_inputs) -> None:
    model = build_model(make_model_config(name=name), make_data_config(target_product="mdn_chla"))
    inputs = make_forecast_inputs()
    assert model.forecast(**inputs).shape == (2, 1, 32, 32)
    assert model.reconstruct(inputs["images"][:, -1], inputs["masks"][:, -1]).shape == (2, 1, 32, 32)


def test_mdn_encoder_forecasts_one_pseudo_label_channel(make_model_config, make_data_config) -> None:
    data_config = make_data_config(target_product="mdn_chla")
    model = build_model(make_model_config(encoder="mdn_spectral"), data_config)
    batch, steps, size = 1, 3, 32
    images = torch.rand(batch, steps, len(data_config.input_bands), size, size) * 0.05
    masks = torch.ones(batch, steps, size, size)
    prediction = model.forecast(images, masks, torch.ones(batch, steps), torch.zeros(batch, steps), torch.zeros(batch))
    assert prediction.shape == (batch, 1, size, size)


def test_persistence_base_is_zero_where_nothing_was_observed() -> None:
    """Unobserved pixels get a finite base, and a valid zero stays zero."""
    values = torch.tensor([[[[[9.0, 9.0]]], [[[0.0, 4.0]]]]])
    masks = torch.tensor([[[[1.0, 1.0]], [[1.0, 0.0]]]])
    frame_valid = torch.tensor([[0.0, 1.0]])
    assert torch.equal(persistence_forecast(values, masks, frame_valid), torch.zeros(1, 1, 1, 2))
