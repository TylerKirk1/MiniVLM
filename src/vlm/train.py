"""Validate local data by default; training requires an explicit --train flag."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ImageQACollator, ImageQADataset
from .runtime import (Settings, load_model, load_processors, read_checkpoint,
                      require_training_device, save_checkpoint)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def check_data(dataset, collator):
    longest = 0
    for batch in DataLoader(dataset, batch_size=1, collate_fn=collator):
        longest = max(longest, batch["input_ids"].shape[1])
        if not batch["pixel_attention_mask"].bool().any(dim=1).all():
            raise ValueError("Image processor returned an image with no valid patches")
        if not torch.isfinite(batch["pixel_values"]).all():
            raise ValueError("Image processor returned non-finite pixels")
    return longest


def train_steps(model, loader, optimizer, steps, accumulation, device):
    """One example per microbatch, averaging over a full accumulation window."""
    model.train()
    iterator = iter(loader)
    parameters = [p for p in model.parameters() if p.requires_grad]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stopping without saving a checkpoint")
            loss.backward()
            mean_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"step {step}/{steps}: loss {mean_loss:.4f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL with image, question, answer fields")
    parser.add_argument("--stage", choices=["alignment", "qlora"], default="alignment")
    parser.add_argument("--init", type=Path, help="Completed alignment checkpoint for QLoRA")
    parser.add_argument("--visual-tokens", type=positive_int)
    parser.add_argument("--max-patches", type=positive_int)
    parser.add_argument("--max-text-tokens", type=positive_int)
    parser.add_argument("--train", action="store_true", help="Explicitly start weight updates")
    parser.add_argument("--steps", type=positive_int, help="Number of optimizer updates")
    parser.add_argument("--accumulation", type=positive_int, default=16)
    parser.add_argument("--output", type=Path, help="New checkpoint directory; never overwritten")
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"Manifest does not exist: {args.manifest}")
    if args.stage == "qlora" and args.init is None:
        parser.error("--stage qlora requires --init pointing to an alignment checkpoint")
    if args.stage == "alignment" and args.init is not None:
        parser.error("--init is only used to start QLoRA from alignment")
    if args.train and (args.steps is None or args.output is None):
        parser.error("--train requires explicit --steps and --output")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new directory")

    settings = Settings()
    if args.init is not None:
        metadata = read_checkpoint(args.init)
        if metadata["stage"] != "alignment":
            parser.error("--init must be an alignment checkpoint; run resumption is not supported")
        settings = Settings(**metadata["settings"])
        if args.visual_tokens is not None and args.visual_tokens != settings.num_latents:
            parser.error("Visual token count must match the alignment checkpoint")
    for field, value in [("num_latents", args.visual_tokens), ("max_patches", args.max_patches),
                         ("max_text_tokens", args.max_text_tokens)]:
        if value is not None:
            setattr(settings, field, value)
    if args.train:
        require_training_device()
    tokenizer, processor = load_processors(settings)
    dataset = ImageQADataset(args.manifest, tokenizer, settings.max_text_tokens)
    collator = ImageQACollator(tokenizer, processor, settings.max_patches)
    longest = check_data(dataset, collator)
    print(f"Validated {len(dataset)} examples; longest text: {longest} tokens; "
          f"visual tokens: {settings.num_latents}; patch limit: {settings.max_patches}")
    if not args.train:
        print("Preflight only: no model weights loaded, no training, no run files written.")
        try:
            require_training_device()
        except RuntimeError as exc:
            print(f"Training environment still needs attention: {exc}")
        else:
            print("CUDA and quantization dependencies found; full-model VRAM fit is unverified.")
        return

    torch.manual_seed(17)
    model = load_model(settings, args.stage, args.init, training=True)
    groups = [{"params": list(model.bridge.parameters()), "lr": 2e-4}]
    adapters = [p for p in model.language.parameters() if p.requires_grad]
    if adapters:
        groups.append({"params": adapters, "lr": 5e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator,
                        generator=torch.Generator().manual_seed(17))
    train_steps(model, loader, optimizer, args.steps, args.accumulation, torch.device("cuda"))
    save_checkpoint(model, settings, args.stage, args.output)
    print(f"Saved {args.stage} checkpoint to {args.output}")


if __name__ == "__main__":
    main()
