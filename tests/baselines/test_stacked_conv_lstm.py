import pytest
import torch

from chla_prediction.baselines.stacked_conv_lstm import StackedConvLSTMForecaster
from chla_prediction.config import ModelConfig


@pytest.fixture
def model(make_data_config) -> StackedConvLSTMForecaster:
    torch.manual_seed(0)
    model_config = ModelConfig(name="stacked_conv_lstm_forecaster", hidden_channels=8, temporal_depth=2)
    return StackedConvLSTMForecaster(model_config, make_data_config(target_product="mdn_chla"))


def test_forecast_shapes_and_finite_output(model, make_forecast_inputs) -> None:
    prediction = model.forecast(**make_forecast_inputs(input_targets=True))
    assert prediction.shape == (2, 1, 32, 32)
    # The readout is an unbounded regression head (MSE on min-max normalized
    # values, as in the source system), so an untrained model carries no
    # range guarantee, only finiteness.
    assert torch.isfinite(prediction).all()


def test_padded_frames_leave_the_state_untouched(model, make_forecast_inputs) -> None:
    inputs = make_forecast_inputs(input_targets=True)
    reference = model.forecast(**inputs)
    padded = {key: value.clone() for key, value in inputs.items()}
    padded["frame_valid"][:, 1] = 0.0
    with_pad = model.forecast(**padded)
    # With frame_valid 0 the padded frame's content may not move the output.
    padded["input_targets"][:, 1] = 99.0
    assert torch.equal(model.forecast(**padded), with_pad)
    assert not torch.equal(with_pad, reference)


def test_invalid_pixels_do_not_affect_the_forecast(model, make_forecast_inputs) -> None:
    inputs = make_forecast_inputs(input_targets=True)
    inputs["input_target_masks"][:, :, :16] = 0.0
    reference = model.forecast(**inputs)
    inputs["input_targets"][:, :, :, :16] = -123.0
    assert torch.equal(model.forecast(**inputs), reference)
