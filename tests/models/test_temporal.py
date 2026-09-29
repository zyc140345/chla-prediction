import pytest
import torch

from chla_prediction.models.forecaster import Forecaster
from chla_prediction.models.registry import build_model
from chla_prediction.models.temporal import days_before_target

TEMPORAL_MODULES = [
    "conv_gru_forecaster",
    "target_relative_attention_forecaster",
    "simvp_forecaster",
    "space_time_transformer_forecaster",
]


@pytest.fixture
def build_forecaster(make_model_config, make_data_config):
    """Build a forecaster of pseudo-label fields over four-frame windows."""
    data_config = make_data_config(target_product="mdn_chla", sequence={"input_window": 4})

    def build(name: str, **model_fields) -> Forecaster:
        return build_model(make_model_config(name=name, temporal_depth=1, **model_fields), data_config)

    return build


def test_days_before_target_accumulates_later_gaps() -> None:
    gaps = torch.tensor([[0.0, 5.0, 10.0]])
    lead = torch.tensor([7.0])
    assert days_before_target(gaps, lead).tolist() == [[22.0, 17.0, 7.0]]


@pytest.mark.parametrize("name", TEMPORAL_MODULES)
def test_forecast_shapes_up_to_the_input_window(name, build_forecaster, make_forecast_inputs) -> None:
    model = build_forecaster(name)
    inputs = make_forecast_inputs()
    prediction = model.forecast(**inputs)
    assert prediction.shape == (2, 1, 32, 32)
    assert torch.isfinite(prediction).all()
    # Inference feeds unpadded sequences, shorter than the window.
    short = {key: value[:, :3] if key != "lead_days" else value for key, value in inputs.items()}
    assert model.forecast(**short).shape == (2, 1, 32, 32)


@pytest.mark.parametrize("name", TEMPORAL_MODULES)
def test_padded_frames_are_ignored(name, build_forecaster, make_forecast_inputs) -> None:
    model = build_forecaster(name).eval()
    inputs = make_forecast_inputs()
    inputs["frame_valid"][:, 0] = 0.0
    inputs["interval_days"][:, 0] = 0.0
    inputs["images"][:, 0] = 0.0
    inputs["masks"][:, 0] = 0.0
    with torch.no_grad():
        reference = model.forecast(**inputs)
        inputs["images"][:, 0] = torch.rand_like(inputs["images"][:, 0])
        inputs["masks"][:, 0] = 1.0
        perturbed = model.forecast(**inputs)
    assert torch.allclose(reference, perturbed, atol=1e-5)


@pytest.mark.parametrize("name", TEMPORAL_MODULES)
def test_time_blind_models_ignore_gaps_and_lead(name, build_forecaster, make_forecast_inputs) -> None:
    inputs = make_forecast_inputs()
    shifted = inputs | {"interval_days": inputs["interval_days"] * 3.0 + 1.0, "lead_days": inputs["lead_days"] + 60.0}
    blind = build_forecaster(name, time_conditioning=False).eval()
    conditioned = build_forecaster(name).eval()
    with torch.no_grad():
        assert torch.equal(blind.forecast(**inputs), blind.forecast(**shifted))
        # The default model does condition on time: the same shift moves the output.
        assert not torch.equal(conditioned.forecast(**inputs), conditioned.forecast(**shifted))


def test_space_time_transformer_pad_multiple_covers_its_token_grid(build_forecaster, make_forecast_inputs) -> None:
    model = build_forecaster("space_time_transformer_forecaster")
    with pytest.raises(ValueError, match="patch size"):
        model.forecast(**make_forecast_inputs(size=36))
    padded = -(-36 // model.pad_multiple) * model.pad_multiple
    assert model.forecast(**make_forecast_inputs(size=padded)).shape[-2:] == (padded, padded)
