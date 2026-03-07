from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .projector import MLPProjector
from .resampler import LearnedTokenResampler


@dataclass(slots=True)
class BridgeConfig:
    vision_feature_dim: int
    latent_dim: int
    num_latents: int
    depth: int
    num_heads: int
    mlp_ratio: int
    projector_hidden_dim: int
    llm_hidden_size: int


class VisionLanguageBridge(nn.Module):
    """Compresses vision features and projects them into the LLM hidden space."""

    def __init__(self, config: BridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.resampler = LearnedTokenResampler(
            input_dim=config.vision_feature_dim,
            latent_dim=config.latent_dim,
            num_latents=config.num_latents,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        )
        self.projector = MLPProjector(
            input_dim=config.latent_dim,
            hidden_dim=config.projector_hidden_dim,
            output_dim=config.llm_hidden_size,
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        latents = self.resampler(vision_features)
        return self.projector(latents)
