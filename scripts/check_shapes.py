from __future__ import annotations

import argparse
from pathlib import Path

from vlm.config import load_merged_config, require_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CPU shape check for the visual bridge")
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        required=True,
        help="Path to a YAML config fragment. Pass multiple times to merge in order.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Dummy batch size for the shape check.",
    )
    parser.add_argument(
        "--source-tokens",
        type=int,
        default=576,
        help="Dummy number of vision tokens coming from the encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_config([Path(path) for path in args.configs])

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Torch is not installed in this environment. Install the 'train' extras on the "
            "workstation before using this script."
        ) from exc

    from vlm.models import BridgeConfig, VisionLanguageBridge

    bridge_config = BridgeConfig(
        vision_feature_dim=require_path(config, "model", "vision", "feature_dim"),
        latent_dim=require_path(config, "model", "resampler", "latent_dim"),
        num_latents=require_path(config, "model", "resampler", "num_latents"),
        depth=require_path(config, "model", "resampler", "depth"),
        num_heads=require_path(config, "model", "resampler", "num_heads"),
        mlp_ratio=require_path(config, "model", "resampler", "mlp_ratio"),
        projector_hidden_dim=require_path(config, "model", "projector", "hidden_dim"),
        llm_hidden_size=require_path(config, "model", "llm", "hidden_size"),
    )

    bridge = VisionLanguageBridge(bridge_config)
    dummy_source = torch.randn(
        args.batch_size,
        args.source_tokens,
        bridge_config.vision_feature_dim,
    )
    projected = bridge(dummy_source)

    print(f"input shape:  {tuple(dummy_source.shape)}")
    print(f"output shape: {tuple(projected.shape)}")


if __name__ == "__main__":
    main()
