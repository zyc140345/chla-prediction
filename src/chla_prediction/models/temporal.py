"""Temporal modules turning a sequence of latent frames into one forecast latent.

Every module maps ``latents [B, T, C, h, w]`` with per-frame acquisition
intervals ``interval_days [B, T]`` (days since the previous frame, 0 for the
first), a per-frame validity flag ``frame_valid [B, T]`` (0 for left-padded
frames) and the forecast lead ``lead_days [B]`` to a hidden map
``[B, hidden, h, w]``.

* ``ConvGRU`` / ``ConvLSTM``: recurrent roll-out, one step per frame, the
  acquisition interval entering as a broadcast channel; the forecast lead
  is only seen by the decoder.
* ``TargetRelativeAttention`` (proposed): per-position attention over the
  observed frames with a continuous-time encoding of the days before the
  target (LTAE-style) and a learned query token at the target time.
* ``SimVPTranslator``: recurrent-free stack of large-kernel gated
  convolutions (SimVP/TAU) over the channel-stacked frames.
* ``SpaceTimeTransformer``: factorized temporal/spatial attention over
  patch tokens (PredFormer/TSViT-style).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

DAYS_SCALE = 30.0
TIME_ENCODING_MAX_DAYS = 400.0

__all__ = [
    "ConvGRU",
    "ConvGRUCell",
    "ConvLSTM",
    "ConvLSTMCell",
    "SimVPTranslator",
    "SpaceTimeTransformer",
    "TargetRelativeAttention",
    "days_before_target",
    "time_encoding",
]


def days_before_target(interval_days: torch.Tensor, lead_days: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` days between each input frame and the target date."""
    later = interval_days.flip(1).cumsum(1).flip(1) - interval_days
    return later + lead_days.view(-1, 1)


def time_encoding(days: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding ``[..., dim]`` of day offsets, wavelengths from 1 day to ``TIME_ENCODING_MAX_DAYS``."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(TIME_ENCODING_MAX_DAYS) * torch.arange(half, device=days.device, dtype=torch.float32) / half
    )
    angles = days.unsqueeze(-1).float() * frequencies * (2 * math.pi)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        channels = input_channels + hidden_channels
        self.gates = nn.Conv2d(channels, hidden_channels * 2, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(channels, hidden_channels, kernel_size, padding=padding)
        self.hidden_channels = hidden_channels

    def forward(self, frame: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([frame, hidden], dim=1)
        update, reset = torch.sigmoid(self.gates(stacked)).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([frame, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell without the peephole terms of Shi et al. (2015, Eq. 3), as in Keras ``ConvLSTM2D``."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, kernel_size, padding=padding)
        self.hidden_channels = hidden_channels

    def forward(
        self, frame: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = self.gates(torch.cat([frame, hidden], dim=1))
        in_gate, forget_gate, out_gate, candidate = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(in_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(out_gate) * torch.tanh(cell)
        return hidden, cell


class _IntervalConditionedRollout(nn.Module):
    """Roll a recurrent cell over the latent frames.

    Each step sees its acquisition interval as a broadcast channel; padded
    frames leave the state untouched.
    """

    spatial_multiple = 1

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(
        self,
        latents: torch.Tensor,
        interval_days: torch.Tensor,
        frame_valid: torch.Tensor,
        lead_days: torch.Tensor,
    ) -> torch.Tensor:
        del lead_days
        batch, steps, _, height, width = latents.shape
        state = self.initial_state(latents)
        for step in range(steps):
            interval = (interval_days[:, step] / DAYS_SCALE).view(batch, 1, 1, 1)
            frame = torch.cat([latents[:, step], interval.expand(batch, 1, height, width)], dim=1)
            update = self.step(frame, state)
            keep = frame_valid[:, step].view(batch, 1, 1, 1)
            state = tuple(keep * new + (1.0 - keep) * old for new, old in zip(update, state, strict=True))
        return state[0]

    def initial_state(self, latents: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def step(self, frame: torch.Tensor, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError


class ConvGRU(_IntervalConditionedRollout):
    def __init__(self, latent_channels: int, hidden_channels: int, **_):
        super().__init__(hidden_channels)
        self.cell = ConvGRUCell(latent_channels + 1, hidden_channels)

    def initial_state(self, latents: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, _, _, height, width = latents.shape
        return (latents.new_zeros((batch, self.hidden_channels, height, width)),)

    def step(self, frame: torch.Tensor, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (self.cell(frame, state[0]),)


class ConvLSTM(_IntervalConditionedRollout):
    """Recurrent roll-out of ``ConvLSTMCell``."""

    def __init__(self, latent_channels: int, hidden_channels: int, **_):
        super().__init__(hidden_channels)
        self.cell = ConvLSTMCell(latent_channels + 1, hidden_channels)

    def initial_state(self, latents: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, _, _, height, width = latents.shape
        zeros = latents.new_zeros((batch, self.hidden_channels, height, width))
        return (zeros, zeros.clone())

    def step(self, frame: torch.Tensor, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return self.cell(frame, (state[0], state[1]))


# --- transformer building blocks -------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention over ``[N, L, D]`` with an optional key mask ``[N, L]`` (True = attend).

    ``short`` computes the softmax explicitly for the temporal axis, where
    ``N`` (batch x positions) exceeds the fused kernels' launch limits on
    full scenes.
    """

    def __init__(self, dim: int, heads: int, short: bool = False):
        super().__init__()
        self.heads = heads
        self.short = short
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        n, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(n, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if self.short:
            scores = query @ key.transpose(-2, -1) / math.sqrt(dim // self.heads)
            if key_mask is not None:
                scores = scores.masked_fill(~key_mask.view(n, 1, 1, length), float("-inf"))
            out = scores.softmax(dim=-1) @ value
        else:
            attn_mask = None if key_mask is None else key_mask.view(n, 1, 1, length)
            out = functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(n, length, dim))


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2, short: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _Attention(dim, heads, short)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, tokens: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens), key_mask)
        return tokens + self.mlp(self.norm2(tokens))


def _conv_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=channels),
        nn.SiLU(inplace=True),
    )


class _FrameConditioning(nn.Module):
    """Add the time encoding of each frame's days before the target to its channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.time = nn.Linear(channels, channels)

    def forward(self, latents: torch.Tensor, interval_days: torch.Tensor, lead_days: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, _, _ = latents.shape
        offsets = self.time(time_encoding(days_before_target(interval_days, lead_days), channels))
        return latents + offsets.view(batch, steps, channels, 1, 1)


class TargetRelativeAttention(nn.Module):
    """Target-relative temporal attention: a per-position temporal transformer with a target-time query token.

    Frame tokens carry the latent plus the encoding of their days before the
    target; padded frames are masked out. A 3x3 convolution block before and
    after the attention stack gives the tokens local spatial context.
    """

    spatial_multiple = 1

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        depth: int = 2,
        heads: int = 4,
        **_,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.project = (
            nn.Identity() if latent_channels == hidden_channels else nn.Conv2d(latent_channels, hidden_channels, 1)
        )
        self.mix_in = _conv_block(hidden_channels)
        self.condition = _FrameConditioning(hidden_channels)
        self.query = nn.Parameter(torch.zeros(hidden_channels))
        self.blocks = nn.ModuleList(_TransformerBlock(hidden_channels, heads, short=True) for _ in range(depth))
        self.norm = nn.LayerNorm(hidden_channels)
        self.mix_out = _conv_block(hidden_channels)

    def forward(
        self,
        latents: torch.Tensor,
        interval_days: torch.Tensor,
        frame_valid: torch.Tensor,
        lead_days: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _, height, width = latents.shape
        flat = self.mix_in(self.project(latents.flatten(0, 1)))
        latents = flat.view(batch, steps, self.hidden_channels, height, width)
        latents = self.condition(latents, interval_days, lead_days)
        # Query token at the target date (zero days before target).
        query = self.query + self.condition.time(time_encoding(latents.new_zeros(()), self.hidden_channels))
        query = query.view(1, 1, self.hidden_channels, 1, 1).expand(batch, 1, -1, height, width)
        tokens = torch.cat([latents, query], dim=1)  # [B, T+1, C, h, w]
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(batch * height * width, steps + 1, self.hidden_channels)
        key_mask = torch.cat([frame_valid > 0, frame_valid.new_ones((batch, 1)) > 0], dim=1)
        key_mask = key_mask.view(batch, 1, 1, steps + 1).expand(batch, height, width, steps + 1).reshape(-1, steps + 1)
        for block in self.blocks:
            tokens = block(tokens, key_mask)
        out = self.norm(tokens[:, -1]).view(batch, height, width, self.hidden_channels).permute(0, 3, 1, 2)
        return self.mix_out(out.contiguous())


class _TAUBlock(nn.Module):
    """Temporal Attention Unit (Tan et al., CVPR 2023).

    Large-kernel depthwise spatial attention gated by squeeze-excitation
    channel attention, then a pointwise MLP; both residual.
    """

    def __init__(self, channels: int, kernel_size: int = 21, dilation: int = 3, mlp_ratio: int = 4):
        super().__init__()
        small = 2 * dilation - 1
        large = kernel_size // dilation
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.static = nn.Sequential(
            nn.Conv2d(channels, channels, small, padding=small // 2, groups=channels),
            nn.Conv2d(channels, channels, large, padding=(large // 2) * dilation, dilation=dilation, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.dynamic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * mlp_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * mlp_ratio, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + h * self.static(h) * self.dynamic(h)
        return x + self.mlp(self.norm2(x))


class SimVPTranslator(nn.Module):
    """Recurrent-free spatiotemporal translator (SimVP v2 / TAU).

    The latent frames with their interval and validity channels are
    stacked along channels, projected to a working width and passed
    through TAU blocks; the forecast lead enters as one more channel.
    """

    spatial_multiple = 1

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        depth: int = 4,
        input_window: int = 6,
        width: int = 256,
        **_,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_window = input_window
        self.frame_channels = latent_channels + 2
        self.stem = nn.Sequential(
            nn.Conv2d(input_window * self.frame_channels + 1, width, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(_TAUBlock(width) for _ in range(depth)))
        self.head = nn.Conv2d(width, hidden_channels, kernel_size=1)

    def forward(
        self,
        latents: torch.Tensor,
        interval_days: torch.Tensor,
        frame_valid: torch.Tensor,
        lead_days: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _, height, width = latents.shape
        if steps > self.input_window:
            raise ValueError(f"SimVPTranslator built for {self.input_window} frames, got {steps}")

        def expand(values: torch.Tensor) -> torch.Tensor:
            return values.view(batch, steps, -1, 1, 1).expand(batch, steps, -1, height, width)

        channels = [latents, expand(interval_days / DAYS_SCALE), expand(frame_valid)]
        frames = torch.cat(channels, dim=2) * frame_valid.view(batch, steps, 1, 1, 1)
        if steps < self.input_window:  # left-pad shorter sequences like the dataset does
            pad = frames.new_zeros((batch, self.input_window - steps, self.frame_channels, height, width))
            frames = torch.cat([pad, frames], dim=1)
        lead = (lead_days / DAYS_SCALE).view(batch, 1, 1, 1).expand(batch, 1, height, width)
        x = torch.cat([frames.flatten(1, 2), lead], dim=1)
        return self.head(self.blocks(self.stem(x)))


def _grid_encoding(height: int, width: int, dim: int, device) -> torch.Tensor:
    """Fixed 2-D sine-cosine position encoding ``[height * width, dim]``."""
    quarter = dim // 4
    frequencies = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    parts = []
    for coordinate in (ys.reshape(-1), xs.reshape(-1)):
        angles = coordinate.unsqueeze(-1) * frequencies
        parts += [angles.sin(), angles.cos()]
    return torch.cat(parts, dim=-1)


class SpaceTimeTransformer(nn.Module):
    """Factorized space-time transformer over patch tokens.

    Latent frames are patchified and a query frame at the target date is
    appended. Each block applies temporal attention within a patch position
    (with the continuous-time encoding and padding mask), then spatial
    attention within a frame. The query frame is unpatchified to the hidden
    map.
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 4,
        dim: int = 192,
        **_,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        self.spatial_multiple = patch_size
        self.dim = dim
        self.embed = nn.Linear(latent_channels * patch_size**2, dim)
        self.condition = _FrameConditioning(dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.temporal = nn.ModuleList(_TransformerBlock(dim, heads, short=True) for _ in range(depth))
        self.spatial = nn.ModuleList(_TransformerBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, hidden_channels * patch_size**2)
        self.mix_out = _conv_block(hidden_channels)

    def forward(
        self,
        latents: torch.Tensor,
        interval_days: torch.Tensor,
        frame_valid: torch.Tensor,
        lead_days: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, channels, height, width = latents.shape
        p = self.patch_size
        if height % p or width % p:
            raise ValueError(f"Latent grid {height}x{width} is not divisible by patch size {p}")
        gh, gw = height // p, width // p
        # Patchify: [B, T, C, gh, p, gw, p] -> [B, T, gh*gw, C*p*p].
        patches = latents.view(batch, steps, channels, gh, p, gw, p).permute(0, 1, 3, 5, 2, 4, 6)
        tokens = self.embed(patches.reshape(batch, steps, gh * gw, channels * p * p))
        tokens = tokens.permute(0, 1, 3, 2).reshape(batch, steps, self.dim, gh, gw)
        tokens = self.condition(tokens, interval_days, lead_days)
        query = self.query + self.condition.time(time_encoding(tokens.new_zeros(()), self.dim))
        tokens = torch.cat([tokens, query.view(1, 1, self.dim, 1, 1).expand(batch, 1, -1, gh, gw)], dim=1)
        tokens = tokens.flatten(3).permute(0, 1, 3, 2)  # [B, T+1, N, D]
        tokens = tokens + _grid_encoding(gh, gw, self.dim, tokens.device).view(1, 1, gh * gw, self.dim)
        n = gh * gw
        key_mask = torch.cat([frame_valid > 0, frame_valid.new_ones((batch, 1)) > 0], dim=1)
        key_mask = key_mask.view(batch, 1, steps + 1).expand(batch, n, steps + 1).reshape(batch * n, steps + 1)
        for temporal, spatial in zip(self.temporal, self.spatial, strict=True):
            tokens = tokens.permute(0, 2, 1, 3).reshape(batch * n, steps + 1, self.dim)
            tokens = temporal(tokens, key_mask)
            tokens = (
                tokens.view(batch, n, steps + 1, self.dim).permute(0, 2, 1, 3).reshape(batch * (steps + 1), n, self.dim)
            )
            tokens = spatial(tokens)
            tokens = tokens.view(batch, steps + 1, n, self.dim)
        out = self.unembed(self.norm(tokens[:, -1]))  # [B, N, hidden*p*p]
        out = out.view(batch, gh, gw, self.hidden_channels, p, p).permute(0, 3, 1, 4, 2, 5)
        return self.mix_out(out.reshape(batch, self.hidden_channels, height, width))
