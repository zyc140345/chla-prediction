# SPDX-License-Identifier: Apache-2.0
# Adapted from the Clay foundation model (Apache-2.0) and vit-pytorch (MIT).
"""Clay v1.5 encoder (frozen base weights, trainable LoRA adapters), vendored minimally.

The architecture code below is adapted from the Clay foundation model
(https://github.com/Clay-foundation/model, Apache-2.0) and from
vit-pytorch's simple ViT / the DOFA dynamic embedding it builds on. Vendoring
avoids the claymodel package's hard torch==2.4 pin. Weight keys match the
released clay-v1.5.ckpt, which lacks only the band-statistics buffers.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

# Sentinel-2 L2A statistics from Clay's configs/metadata.yaml: DN-scale mean
# and std plus wavelengths (um). Clay's metadata has no coastal-aerosol
# band; B1 reuses blue-band-like statistics with its own wavelength, which
# the wave-aware patch embedding consumes.
CLAY_S2_STATS = {
    "B1": (1105.0, 1809.0, 0.443),
    "B2": (1105.0, 1809.0, 0.493),
    "B3": (1355.0, 1757.0, 0.56),
    "B4": (1552.0, 1888.0, 0.665),
    "B5": (1887.0, 1870.0, 0.704),
    "B6": (2422.0, 1732.0, 0.74),
    "B7": (2630.0, 1697.0, 0.783),
    "B8": (2743.0, 1742.0, 0.842),
    "B8A": (2785.0, 1648.0, 0.865),
    "B11": (2388.0, 1470.0, 1.61),
    "B12": (1835.0, 1379.0, 2.19),
}
DN_SCALE = 10000.0
GSD = 10.0

__all__ = ["ClayFrozenEncoder"]


def posemb_sincos_2d_with_gsd(h, w, dim, gsd, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).type(dtype)


def posemb_sincos_1d(waves, dim, temperature=10000, dtype=torch.float32):
    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)
    scaled = waves[:, None] * omega[None, :]
    return torch.cat((scaled.sin(), scaled.cos()), dim=1).type(dtype)


class FCBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = functional.gelu(self.l1(x))
        y = functional.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(self, wave_dim, output_dim, num_latent_tokens, embed_dim):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim, nhead=4, activation="gelu", dropout=0, norm_first=False, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = nn.Linear(wave_dim, embed_dim)
        self.weight_tokens = nn.Parameter(torch.randn(num_latent_tokens, wave_dim) * 0.02)
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1])
        bias = self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(self, wave_dim, num_latent_tokens, patch_size, embed_dim):
        super().__init__()
        self.wave_dim = wave_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.weight_generator = WavesTransformer(wave_dim, (patch_size**2) * embed_dim, num_latent_tokens, embed_dim)
        self.fclayer = FCBlock(wave_dim)

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim).to(batch.device)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)
        dynamic_weight = weight.reshape(-1, self.embed_dim, self.patch_size, self.patch_size).permute(1, 0, 2, 3)
        out = functional.conv2d(batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size)
        return out.flatten(2).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        batch, tokens, _ = x.shape
        q, k, v = (t.view(batch, tokens, self.heads, -1).transpose(1, 2) for t in qkv)
        x = functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [nn.ModuleList([Attention(dim, heads, dim_head), FeedForward(dim, mlp_dim)]) for _ in range(depth)]
        )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ClayFrozenEncoder(nn.Module):
    """Clay v1.5 encoder with the base weights frozen.

    Consumes reflectance frames plus validity masks (invalid pixels are
    filled with the band mean, i.e. zero after normalization) and returns a
    spatial latent map at 1/8 resolution. Low-rank adapters on the attention
    projections (``lora_rank``) stay trainable for water-domain continued
    pretraining; ``lora_rank=0`` freezes the whole encoder.
    """

    def __init__(
        self,
        band_names: list[str],
        checkpoint: Path | None,
        dim: int = 1024,
        depth: int = 24,
        heads: int = 16,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.latent_channels = dim
        self.downsample_factor = patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.patch_embedding = DynamicEmbedding(
            wave_dim=128, num_latent_tokens=128, patch_size=patch_size, embed_dim=dim
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, int(dim * mlp_ratio))

        stats = torch.tensor([CLAY_S2_STATS[band] for band in band_names])
        self.register_buffer("band_mean", stats[:, 0].view(1, -1, 1, 1))
        self.register_buffer("band_std", stats[:, 1].view(1, -1, 1, 1))
        self.register_buffer("waves", stats[:, 2])

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
        self.requires_grad_(False)
        if lora_rank > 0:
            from peft import LoraConfig, inject_adapter_in_model

            lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, target_modules=["to_qkv", "to_out"])
            inject_adapter_in_model(lora_config, self.transformer)
        self.trainable = lora_rank > 0

    def _load_checkpoint(self, checkpoint: Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        prefix = "model.encoder."
        encoder_state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        # The band statistics and wavelengths are buffers set in __init__, not
        # checkpoint keys; any other missing or unexpected key is an error.
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        missing = [k for k in missing if not k.startswith("band_") and k != "waves"]
        if missing or unexpected:
            raise ValueError(f"Clay checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=frames.is_cuda)
        grad_mode = torch.enable_grad() if (self.trainable and self.training) else torch.no_grad()
        with grad_mode, autocast:
            valid = masks.unsqueeze(1)
            dn = frames * DN_SCALE * valid + self.band_mean * (1.0 - valid)
            x = (dn - self.band_mean) / self.band_std

            patches = self.patch_embedding(x, self.waves)
            batch, length, _ = patches.shape
            grid_h = x.shape[-2] // self.patch_size
            grid_w = x.shape[-1] // self.patch_size
            pos = posemb_sincos_2d_with_gsd(grid_h, grid_w, self.dim - 8, gsd=torch.tensor(GSD)).to(patches.device)
            metadata = patches.new_zeros((batch, length, 8))
            patches = patches + torch.cat((pos.unsqueeze(0).expand(batch, -1, -1), metadata), dim=-1)

            tokens = torch.cat((self.cls_token.expand(batch, -1, -1), patches), dim=1)
            encoded = self.transformer(tokens)[:, 1:]
            return encoded.transpose(1, 2).reshape(batch, self.dim, grid_h, grid_w).float()
