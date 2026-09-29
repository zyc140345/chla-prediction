import torch

from chla_prediction.baselines.reference import ClimatologyForecaster, PersistenceForecaster
from chla_prediction.config import ModelConfig


def test_persistence_takes_the_most_recent_valid_pixel(make_data_config) -> None:
    data_config = make_data_config(output_bands=["B4", "B5"])
    model = PersistenceForecaster(ModelConfig(name="persistence_forecaster"), data_config)
    batch, steps, channels, size = 1, 3, len(data_config.input_bands), 4
    images = torch.zeros(batch, steps, channels, size, size)
    images[:, 0] = 1.0
    images[:, 1] = 2.0
    images[:, 2] = 3.0
    masks = torch.ones(batch, steps, size, size)
    # The last frame is cloudy at pixel (0, 0): persistence must fall back to
    # the second frame there and use the last frame elsewhere.
    masks[:, 2, 0, 0] = 0.0
    masks[:, :, 0, 1] = 0.0
    output_indices = [data_config.input_bands.index(b) for b in data_config.output_bands]
    prediction = model.forecast(
        images,
        masks,
        torch.ones(batch, steps),
        torch.zeros(batch, steps),
        torch.zeros(batch),
        images[:, :, output_indices],
        masks,
    )
    assert prediction.shape == (batch, len(output_indices), size, size)
    assert prediction[0, 0, 0, 0].item() == 2.0
    assert prediction[0, 0, 1, 1].item() == 3.0
    assert torch.isnan(prediction[0, :, 0, 1]).all()


def test_persistence_ignores_padded_observations_and_preserves_valid_zero(make_data_config) -> None:
    model = PersistenceForecaster(ModelConfig(name="persistence_forecaster"), make_data_config())
    values = torch.tensor([[[[[9.0, 9.0]]], [[[0.0, 4.0]]]]])
    masks = torch.tensor([[[[1.0, 1.0]], [[1.0, 0.0]]]])
    frame_valid = torch.tensor([[0.0, 1.0]])
    prediction = model.forecast(values, masks, frame_valid, torch.zeros(1, 2), torch.zeros(1), values, masks)
    assert prediction[0, 0, 0, 0] == 0.0
    assert torch.isnan(prediction[0, 0, 0, 1])


def test_persistence_forecasts_pseudo_labels_as_one_channel(make_data_config) -> None:
    model = PersistenceForecaster(
        ModelConfig(name="persistence_forecaster"), make_data_config(input_product="ndci_chla")
    )
    frames = torch.rand(1, 3, 1, 32, 32)
    masks = torch.ones(1, 3, 32, 32)
    prediction = model.forecast(frames, masks, torch.ones(1, 3), torch.zeros(1, 3), torch.zeros(1), frames, masks)
    assert prediction.shape == (1, 1, 32, 32)


def test_climatology_forecaster_returns_the_base_with_undefined_pixels(make_data_config) -> None:
    data_config = make_data_config(input_product="mdn_chla", climatology_base=True)
    model = ClimatologyForecaster(ModelConfig(name="climatology_forecaster"), data_config)
    base = torch.tensor([[[[1.0, float("nan")], [0.5, 2.0]]]])
    out = model.forecast(torch.zeros(1, 2, 1, 2, 2), base=base)
    assert torch.equal(torch.isnan(out), torch.isnan(base))
    assert out[0, 0, 1, 1] == 2.0
