from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, source_dim: int, num_heads: int) -> None:
        super().__init__()
        if num_heads < 1 or latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        self.latents_norm = nn.LayerNorm(latent_dim)
        self.source_norm = nn.LayerNorm(source_dim)

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(source_dim, latent_dim)
        self.value = nn.Linear(source_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, latents: torch.Tensor, source: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, num_latents, latent_dim = latents.shape
        _, num_source_tokens, _ = source.shape

        latents_norm = self.latents_norm(latents)
        source_norm = self.source_norm(source)

        query = self.query(latents_norm)
        key = self.key(source_norm)
        value = self.value(source_norm)

        query = query.view(batch_size, num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_source_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_source_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_mask = None if mask is None else mask[:, None, None, :]
        attention_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, num_latents, latent_dim)
        return latents + self.out(attention_output)


class LearnedTokenResampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_latents: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if min(input_dim, latent_dim, num_latents, depth, num_heads, mlp_ratio) < 1:
            raise ValueError("Resampler dimensions and depth must be positive")

        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            latent_dim=latent_dim,
                            source_dim=input_dim,
                            num_heads=num_heads,
                        ),
                        FeedForward(dim=latent_dim, mlp_ratio=mlp_ratio),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, source: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if source.ndim != 3:
            raise ValueError("Expected source features with shape [batch, tokens, dim]")
        if source.shape[1] == 0:
            raise ValueError("At least one source token is required")
        if mask is not None:
            if mask.shape != source.shape[:2]:
                raise ValueError("Mask must have shape [batch, tokens]")
            mask = mask.to(device=source.device, dtype=torch.bool)
            if not mask.any(dim=1).all():
                raise ValueError("Each image must have at least one unmasked patch")

        batch_size = source.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for attention, feedforward in self.blocks:
            latents = attention(latents, source, mask)
            latents = feedforward(latents)

        return self.output_norm(latents)
