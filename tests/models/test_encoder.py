import numpy as np
import pytest
import torch

from chla_prediction.config import ALL_BANDS
from chla_prediction.models.encoder import MDNSpectralEncoder
from chla_prediction.models.registry import build_model


def _write_mdn_export(path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """An MDN export of one ensemble member with five random 100-wide hidden layers."""
    widths = [7, 100, 100, 100, 100, 100]
    export = {
        "n_rounds": 1,
        "round0_n_targets": 1,
        "round0_n_mix": 5,
        "round0_x_center": rng.random(7).astype(np.float32),
        "round0_x_scale": rng.random(7).astype(np.float32) + 0.5,
        "round0_y_min": np.float32(0.0),
        "round0_y_scale": np.float32(1.0),
    }
    for i in range(5):
        export[f"round0_w{2 * i}"] = rng.normal(size=(widths[i], widths[i + 1])).astype(np.float32)
        export[f"round0_w{2 * i + 1}"] = rng.normal(size=widths[i + 1]).astype(np.float32)
    np.savez(path, **export)
    return export


def test_mdn_spectral_encoder_starts_from_the_mdn(tmp_path) -> None:
    """The spectral stage copies the MDN's five hidden layers and scaler.

    It takes Rrs = reflectance / pi of B1-B7 and stays trainable.
    """
    export = _write_mdn_export(tmp_path / "mdn.npz", np.random.default_rng(0))
    encoder = MDNSpectralEncoder(ALL_BANDS, latent_channels=16, mdn_weights=tmp_path / "mdn.npz")

    frames = torch.rand(2, len(ALL_BANDS), 8, 8) * 0.05
    masks = torch.ones(2, 8, 8)
    masks[0, 0, 0] = 0
    spectral = []
    encoder.spectral.register_forward_hook(lambda module, inputs, output: spectral.append(output.detach().clone()))
    encoder(frames, masks)
    center = torch.from_numpy(export["round0_x_center"]).view(1, 7, 1, 1)
    scale = torch.from_numpy(export["round0_x_scale"]).view(1, 7, 1, 1)
    hidden = (frames[:, :7] / torch.pi - center) / scale * masks.unsqueeze(1)
    for i in range(5):
        weight = torch.from_numpy(export[f"round0_w{2 * i}"])
        bias = torch.from_numpy(export[f"round0_w{2 * i + 1}"]).view(1, -1, 1, 1)
        hidden = torch.relu(torch.einsum("bihw,io->bohw", hidden, weight) + bias)
    # Masked pixels carry no information downstream.
    assert torch.allclose(spectral[0] * masks.unsqueeze(1), hidden * masks.unsqueeze(1), atol=1e-4)
    assert all(parameter.requires_grad for parameter in encoder.spectral.parameters())


@pytest.mark.parametrize("encoder", ["resnet", "resnet50", "convnext"])
def test_torchvision_encoder_forecast_and_reconstruction_shapes(
    encoder, make_model_config, make_data_config, make_forecast_inputs
) -> None:
    model_config = make_model_config(encoder=encoder, temporal_depth=1)
    model = build_model(model_config, make_data_config(target_product="mdn_chla"))
    inputs = make_forecast_inputs()
    prediction = model.forecast(**inputs)
    assert prediction.shape == (2, 1, 32, 32)
    assert torch.isfinite(prediction).all()
    assert model.reconstruct(inputs["images"][:, -1], inputs["masks"][:, -1]).shape == (2, 1, 32, 32)
