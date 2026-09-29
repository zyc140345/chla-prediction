import torch

from chla_prediction.config import ALL_BANDS, ModelConfig
from chla_prediction.models.clay import ClayFrozenEncoder
from chla_prediction.models.registry import build_model


def _small_encoder(lora_rank: int) -> ClayFrozenEncoder:
    return ClayFrozenEncoder(
        band_names=list(ALL_BANDS), checkpoint=None, dim=16, depth=2, heads=2, dim_head=8, lora_rank=lora_rank
    )


def test_randomly_initialized_encoder_is_frozen_and_follows_the_scene_shape() -> None:
    encoder = _small_encoder(lora_rank=0)
    latent = encoder(torch.rand(2, len(ALL_BANDS), 32, 32), torch.ones(2, 32, 32))
    assert latent.shape == (2, 16, 4, 4)
    assert all(not p.requires_grad for p in encoder.parameters())
    # Full scenes are rectangular; the position encoding must follow.
    rect = encoder(torch.rand(1, len(ALL_BANDS), 32, 48), torch.ones(1, 32, 48))
    assert rect.shape == (1, 16, 4, 6)


def test_lora_adapters_are_the_only_trainable_parameters_and_get_gradients() -> None:
    encoder = _small_encoder(lora_rank=4)
    trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora" in name for name in trainable)
    encoder.train()
    out = encoder(torch.rand(1, len(ALL_BANDS), 32, 32), torch.ones(1, 32, 32))
    out.sum().backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_clay_forecaster_trains_encoder_adapters_and_the_other_stages(make_data_config, make_forecast_inputs) -> None:
    model_config = ModelConfig(
        encoder="clay_frozen", clay_dim=16, clay_depth=2, clay_heads=2, base_channels=8, hidden_channels=32
    )
    model = build_model(model_config, make_data_config(target_product="mdn_chla"))
    prediction = model.forecast(**make_forecast_inputs(batch=1, steps=3))
    assert prediction.shape == (1, 1, 32, 32)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert {"encoder", "temporal", "decoder", "latent_project"} <= {name.split(".")[0] for name in trainable}
    assert all("lora" in name for name in trainable if name.startswith("encoder."))
