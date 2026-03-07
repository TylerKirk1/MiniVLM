from __future__ import annotations

import argparse
from pathlib import Path

from vlm.data import validate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a JSONL multimodal dataset manifest")
    parser.add_argument("manifest", type=Path, help="Path to a JSONL manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_jsonl(args.manifest)

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(f"{args.manifest}: ok")


if __name__ == "__main__":
    main()
