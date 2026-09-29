"""Inference of the MDN Chl-a retrieval for Sentinel-2 MSI, in PyTorch.

An independent inference-only implementation of the mixture density network
of Pahlevan et al. (2020, RSE 240:111604). It loads weights exported from
the reference implementation (https://github.com/BrandonSmithJ/MDN,
GPL-3.0), which are not part of this repository and keep its terms.

The published MSI ``chl`` model is an ensemble of ten mixture density
networks over seven Rrs bands (443-783 nm); each network is RobustScaler ->
5 x Dense(100, ReLU) -> Dense(15) whose outputs split into five mixture
priors, means and scales. The point estimate is the mean of the most
probable component, mapped back through the ``MinMax(-1, 1) . log`` target
scaling, and the ensemble estimate is the median over rounds. Export the
weights with ``tools/data/export_mdn_weights.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["MDN_BANDS", "MDN", "read_mdn_export", "surface_reflectance_to_rrs"]

# Sentinel-2 bands feeding the MSI model, in wavelength order.
MDN_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]


def surface_reflectance_to_rrs(reflectance: torch.Tensor) -> torch.Tensor:
    """Remote-sensing reflectance (1/sr) approximated as L2A surface reflectance / pi.

    Sen2Cor output is not water-leaving reflectance; the pseudo-labels inherit that error.
    """
    return reflectance / torch.pi


def read_mdn_export(path: Path) -> list[dict]:
    """Ensemble members of an exported MDN: dense ``weights`` (kernel, bias, ...) and the scaling parameters."""
    export = np.load(path)
    members = []
    for r in range(int(export["n_rounds"])):
        weights = []
        while f"round{r}_w{len(weights)}" in export:
            weights.append(export[f"round{r}_w{len(weights)}"])
        if int(export[f"round{r}_n_targets"]) != 1:
            raise ValueError(f"{path}: expected a single-target (chl) MDN")
        members.append(
            {
                "weights": weights,
                **{key: export[f"round{r}_{key}"] for key in ("x_center", "x_scale", "y_min", "y_scale")},
                "n_mix": int(export[f"round{r}_n_mix"]),
            }
        )
    return members


def _dense(kernel: np.ndarray, bias: np.ndarray) -> nn.Linear:
    linear = nn.Linear(*kernel.shape)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(kernel.T.copy()))
        linear.bias.copy_(torch.from_numpy(bias.copy()))
    return linear


class _EnsembleMember(nn.Module):
    """RobustScaler, the dense stack, and the mean of the most probable mixture component, unscaled."""

    def __init__(self, weights: list[np.ndarray], x_center, x_scale, y_min, y_scale, n_mix: int):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(0, len(weights) - 2, 2):
            layers += [_dense(weights[i], weights[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers, _dense(weights[-2], weights[-1]))
        self.n_mix = n_mix
        self.register_buffer("x_center", torch.as_tensor(x_center, dtype=torch.float32))
        self.register_buffer("x_scale", torch.as_tensor(x_scale, dtype=torch.float32))
        self.register_buffer("y_min", torch.as_tensor(y_min, dtype=torch.float32))
        self.register_buffer("y_scale", torch.as_tensor(y_scale, dtype=torch.float32))

    def forward(self, rrs: torch.Tensor) -> torch.Tensor:
        out = self.mlp((rrs - self.x_center) / self.x_scale)
        prior = out[:, : self.n_mix]
        mu = out[:, self.n_mix : 2 * self.n_mix]
        top = mu.gather(1, prior.argmax(dim=1, keepdim=True)).squeeze(1)
        return torch.exp((top - self.y_min) / self.y_scale)


class MDN(nn.Module):
    """The pretrained MSI Chl-a MDN, frozen: ``[N, 7]`` Rrs -> ``[N]`` Chl-a in mg/m^3."""

    def __init__(self, weights_path: Path):
        super().__init__()
        self.rounds = nn.ModuleList(_EnsembleMember(**member) for member in read_mdn_export(weights_path))
        self.requires_grad_(False)
        self.eval()

    def forward(self, rrs: torch.Tensor) -> torch.Tensor:
        """Median Chl-a of the ensemble."""
        estimates = torch.stack([member(rrs) for member in self.rounds]).sort(dim=0).values
        # np.median semantics: the mean of the two middle values for an even
        # ensemble (torch.median would take the lower one).
        n = estimates.shape[0]
        return 0.5 * (estimates[(n - 1) // 2] + estimates[n // 2])

    @torch.no_grad()
    def chla_map(self, reflectance: torch.Tensor, valid: torch.Tensor, chunk: int = 1 << 18) -> torch.Tensor:
        """Chl-a of a reflectance stack ``[7, H, W]`` (``MDN_BANDS`` order) at ``valid`` pixels.

        NaN elsewhere and where any band is nonpositive.
        """
        _, height, width = reflectance.shape
        flat = reflectance.reshape(reflectance.shape[0], -1).T
        usable = valid.reshape(-1).bool() & (flat > 0).all(dim=1)
        rrs = surface_reflectance_to_rrs(flat[usable])
        out = torch.full((height * width,), float("nan"), dtype=torch.float32, device=reflectance.device)
        pieces = [self(rrs[i : i + chunk]) for i in range(0, rrs.shape[0], chunk)]
        if pieces:
            out[usable] = torch.cat(pieces)
        return out.reshape(height, width)
