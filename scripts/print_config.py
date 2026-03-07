from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm.config import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a merged VLM config")
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        required=True,
        help="Path to a YAML config fragment. Pass multiple times to merge in order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = [Path(path) for path in args.configs]
    merged = load_merged_config(config_paths)
    print(json.dumps(merged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
