"""Check the standalone visual bridge with synthetic features on CPU."""

import argparse

import torch

from vlm.models import BridgeConfig, VisionLanguageBridge


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--source-tokens", type=positive_int, default=64)
    args = parser.parse_args()

    torch.manual_seed(17)
    # Original provisional dimensions; this does not load or validate checkpoints.
    config = BridgeConfig(
        vision_feature_dim=1152,
        latent_dim=1024,
        num_latents=256,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
        projector_hidden_dim=2048,
        llm_hidden_size=2048,
    )
    bridge = VisionLanguageBridge(config)
    source = torch.randn(args.batch_size, args.source_tokens, config.vision_feature_dim)
    projected = bridge(source)
    expected = (args.batch_size, config.num_latents, config.llm_hidden_size)
    if tuple(projected.shape) != expected or not torch.isfinite(projected).all():
        raise RuntimeError(f"Expected finite output with shape {expected}")

    projected.square().mean().backward()
    for name, parameter in bridge.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")

    print(f"input shape:  {tuple(source.shape)}")
    print(f"output shape: {tuple(projected.shape)}")
    print("backward pass: finite gradients")


if __name__ == "__main__":
    main()
