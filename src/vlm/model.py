"""Frozen SigLIP vision features prepended to Qwen text embeddings."""

import torch
from torch import nn

from .models import BridgeConfig, VisionLanguageBridge


class MiniVLM(nn.Module):
    def __init__(self, vision, language, bridge_config: BridgeConfig):
        super().__init__()
        self.vision = vision.requires_grad_(False).eval()
        self.language = language
        self.bridge = VisionLanguageBridge(bridge_config)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    def embed(self, pixel_values, pixel_attention_mask, spatial_shapes, input_ids, attention_mask):
        with torch.no_grad():
            features = self.vision(
                pixel_values=pixel_values.to(dtype=self.vision.dtype),
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            ).last_hidden_state
        # The connector keeps FP32 parameters even when the frozen towers use BF16/4-bit.
        visual = self.bridge(features.float(), pixel_attention_mask)
        text = self.language.get_input_embeddings()(input_ids)
        embeddings = torch.cat((visual.to(text.dtype), text), dim=1)
        prefix_mask = attention_mask.new_ones((input_ids.shape[0], visual.shape[1]))
        mask = torch.cat((prefix_mask, attention_mask), dim=1)
        if embeddings.shape[1] > self.language.config.max_position_embeddings:
            raise ValueError("Visual tokens plus text exceed the language context limit")
        return embeddings, mask

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes,
                input_ids, attention_mask, labels):
        embeddings, mask = self.embed(
            pixel_values, pixel_attention_mask, spatial_shapes, input_ids, attention_mask
        )
        prefix = labels.new_full((labels.shape[0], self.bridge.config.num_latents), -100)
        targets = torch.cat((prefix, labels), dim=1)
        positions = (mask.cumsum(dim=1) - 1).masked_fill(mask == 0, 0)
        return self.language(
            inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
            labels=targets, use_cache=False,
        )

    @torch.no_grad()
    def generate(self, pixel_values, pixel_attention_mask, spatial_shapes,
                 input_ids, attention_mask, max_new_tokens=64):
        if input_ids.shape[0] != 1 or not attention_mask.all():
            raise ValueError("Generation expects one unpadded question")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        embeddings, mask = self.embed(
            pixel_values, pixel_attention_mask, spatial_shapes, input_ids, attention_mask
        )
        if embeddings.shape[1] + max_new_tokens > self.language.config.max_position_embeddings:
            raise ValueError("Requested answer exceeds the language context limit")
        return self.language.generate(
            inputs_embeds=embeddings, attention_mask=mask,
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
