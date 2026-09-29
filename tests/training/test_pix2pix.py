import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from chla_prediction.training import pix2pix as training
from chla_prediction.training.objectives import masked_l1


class _TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 1, 1)

    def forward(self, condition, image):
        return self.conv(torch.cat([condition, image], dim=1))


class _TinyForecaster(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.generator = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Dropout(0.2), nn.Tanh())


def _tiny_loaders(config):
    rng = torch.Generator().manual_seed(91)
    entries = [
        {
            "input_targets": torch.rand(3, 1, 4, 4, generator=rng),
            "input_target_masks": torch.ones(3, 4, 4),
            "target": torch.rand(1, 4, 4, generator=rng),
            "target_valid": torch.ones(4, 4),
        }
        for _ in range(6)
    ]
    loaders = {
        "forecast": DataLoader(entries, batch_size=2, shuffle=True),
        "forecast_val": DataLoader(entries, batch_size=2),
    }
    return loaders, {"train_sequences": 6, "val_sequences": 6}


def test_gan_step_uses_bce_and_both_preupdate_gradients() -> None:
    torch.manual_seed(7)
    g, d = nn.Conv2d(3, 3, 1), _TinyDiscriminator()
    condition, real = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    valid = torch.ones_like(real)
    fake = g(condition)
    sr, sf = d(condition, real), d(condition, fake)
    lg = F.softplus(-sf).mean() + 100 * masked_l1(fake, real, valid)
    ld = F.softplus(-sr).mean() + F.softplus(sf).mean()
    expected_g = torch.autograd.grad(lg, tuple(g.parameters()), retain_graph=True)
    expected_d = torch.autograd.grad(ld, tuple(d.parameters()))
    # SGD makes the shared-gradient update directly observable in parameters.
    initial_g = [p.detach().clone() for p in g.parameters()]
    initial_d = [p.detach().clone() for p in d.parameters()]
    optim = torch.optim.SGD(g.parameters(), lr=0.01), torch.optim.SGD(d.parameters(), lr=0.01)
    _, stats = training._shared_update(g, d, condition, real, valid, optim)
    assert abs(stats["discriminator"] - float(ld.detach())) < 1e-6
    for initial, gradient, actual in zip(
        initial_g + initial_d, expected_g + expected_d, tuple(g.parameters()) + tuple(d.parameters()), strict=True
    ):
        torch.testing.assert_close(actual, initial - 0.01 * gradient)


def test_training_resumes_exactly_at_an_epoch_boundary(tmp_path, monkeypatch, make_data_config, write_yaml) -> None:
    monkeypatch.setattr(training, "Pix2PixForecaster", _TinyForecaster)
    monkeypatch.setattr(training, "PatchDiscriminator", _TinyDiscriminator)
    monkeypatch.setattr(training, "build_loaders", _tiny_loaders)
    config = {
        "output_dir": str(tmp_path),
        "data": make_data_config(target_product="mdn_chla").model_dump(mode="json"),
        "objective": {"reconstruction_weight": 0.0},
        "phases": [{"name": "forecast", "objective": "forecast", "epochs": 2}],
        "optim": {"device": "cpu"},
    }
    training.train_pix2pix(write_yaml(tmp_path / "continuous.yaml", config | {"run_name": "continuous"}))
    expected = torch.load(tmp_path / "continuous/checkpoint_last.pt", weights_only=True)

    interrupted = write_yaml(tmp_path / "interrupted.yaml", config | {"run_name": "interrupted"})
    original_epoch = training._epoch
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Train (1) and validation (2) of epoch 0, then die in epoch 1.
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(training, "_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train_pix2pix(interrupted)
    monkeypatch.setattr(training, "_epoch", original_epoch)
    training.train_pix2pix(interrupted, resume=True)
    actual = torch.load(tmp_path / "interrupted/checkpoint_last.pt", weights_only=True)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
