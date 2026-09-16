"""Answer one image question using a saved MiniVLM checkpoint."""

import argparse
from pathlib import Path

import torch

from .data import open_image, prompt_ids
from .runtime import Settings, load_model, load_processors, read_checkpoint
from .train import positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Question must not be empty")
    metadata = read_checkpoint(args.checkpoint)
    settings = Settings(**metadata["settings"])
    tokenizer, processor = load_processors(settings)
    ids = prompt_ids(tokenizer, args.question)
    if len(ids) + args.max_new_tokens > settings.max_text_tokens:
        parser.error("Question and answer budget exceed this checkpoint's text token limit")
    inputs = processor(images=[open_image(args.image)], max_num_patches=settings.max_patches,
                       return_tensors="pt")
    inputs["input_ids"] = torch.tensor([ids], dtype=torch.long)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    model = load_model(settings, metadata["stage"], args.checkpoint)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(**{key: value.to("cuda") for key, value in inputs.items()},
                                   max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
