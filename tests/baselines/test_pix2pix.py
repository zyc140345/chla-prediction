import copy

import torch
from torch import nn
from torch.nn import functional as F

from chla_prediction.baselines import pix2pix
from chla_prediction.config import ModelConfig


def _generator_outputs(model: pix2pix.UNetGenerator, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Outputs in eval mode under dropout seeds 43, 43 and 44, then in training mode under seed 43."""
    outputs = {}
    with torch.no_grad():
        model.eval()
        for name, seed in [("first", 43), ("repeated", 43), ("different", 44)]:
            torch.manual_seed(seed)
            outputs[name] = model(image)
        model.train()
        torch.manual_seed(43)
        outputs["training"] = model(image)
    return outputs


def test_generator_keeps_dropout_and_current_batch_statistics_at_inference() -> None:
    torch.manual_seed(42)
    outputs = _generator_outputs(pix2pix.UNetGenerator(width=2), torch.randn(1, 3, 256, 256))
    assert torch.equal(outputs["first"], outputs["repeated"])
    assert torch.equal(outputs["first"], outputs["training"])
    assert not torch.equal(outputs["first"], outputs["different"])
    assert torch.isfinite(outputs["first"]).all()


def test_generator_forward_leaves_its_state_unchanged() -> None:
    torch.manual_seed(42)
    model = pix2pix.UNetGenerator(width=2)
    before = copy.deepcopy(model.state_dict())
    _generator_outputs(model, torch.randn(1, 3, 256, 256))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_current_batch_norm_of_a_single_value_is_zero_and_trains_only_the_bias() -> None:
    bn = pix2pix.CurrentBatchNorm(2)
    x = torch.randn(1, 2, 1, 1, requires_grad=True)
    y = bn(x)
    y.sum().backward()
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert torch.equal(bn.bias.grad, torch.ones_like(bn.bias))


def test_resize_inference_uses_training_gap_fill_order(monkeypatch, make_data_config) -> None:
    monkeypatch.setattr(pix2pix, "UNetGenerator", nn.Identity)
    data = make_data_config(target_product="mdn_chla", spatial_sampling="resize")
    model = pix2pix.Pix2PixForecaster(ModelConfig(), data)
    values = torch.zeros(1, 3, 1, 512, 512)
    values[:, :, :, 1::2] = 2.0
    masks = torch.ones(1, 3, 512, 512)
    masks[:, :, 256:, 256:] = 0.0
    resized_values = F.interpolate(values.reshape(1, 3, 512, 512), (256, 256), mode="nearest-exact")[:, :, None]
    resized_masks = F.interpolate(masks, (256, 256), mode="nearest-exact")
    expected_pack = pix2pix.pack_frames(resized_values, resized_masks)
    expected = F.interpolate(pix2pix.from_tanh(expected_pack[:, -1:]), (512, 512), mode="nearest-exact")
    prediction = model.forecast(values, masks, torch.ones(1, 3), torch.zeros(1, 3), torch.zeros(1), values, masks)
    torch.testing.assert_close(prediction, expected)
