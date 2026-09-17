"""Single-image, single-turn examples; no silent truncation of OCR targets."""

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def open_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def encode_example(tokenizer, question, answer, max_text_tokens):
    prompt = prompt_ids(tokenizer, question)
    target = tokenizer.encode(answer, add_special_tokens=False)
    if not target or tokenizer.eos_token_id is None:
        raise ValueError("An answer and an EOS token are required")
    target.append(tokenizer.eos_token_id)
    if len(prompt) + len(target) > max_text_tokens:
        raise ValueError(
            f"Example needs {len(prompt) + len(target)} text tokens; limit is "
            f"{max_text_tokens}. Shorten it or raise --max-text-tokens."
        )
    return prompt + target, [-100] * len(prompt) + target


class ImageQADataset(Dataset):
    def __init__(self, manifest, tokenizer, max_text_tokens=512):
        self.manifest = Path(manifest).resolve()
        self.examples = []
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or set(row) != {"image", "question", "answer"}:
                        raise ValueError("Expected exactly image, question, and answer fields")
                    if any(not isinstance(v, str) or not v.strip() for v in row.values()):
                        raise ValueError("All fields must be nonempty strings")
                    path = (self.manifest.parent / row["image"]).resolve()
                    if not path.is_file():
                        raise ValueError(f"Image does not exist: {path}")
                    ids, labels = encode_example(
                        tokenizer, row["question"], row["answer"], max_text_tokens
                    )
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{self.manifest}:{line_number}: {exc}") from exc
                self.examples.append((path, ids, labels))
        if not self.examples:
            raise ValueError(f"No examples in {self.manifest}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, ids, labels = self.examples[index]
        try:
            image = open_image(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode image {path}: {exc}") from exc
        return {"image": image, "input_ids": ids, "labels": labels}


class ImageQACollator:
    def __init__(self, tokenizer, image_processor, max_patches=256):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define a padding token")
        self.image_processor = image_processor
        self.max_patches = max_patches

    def __call__(self, examples):
        images = self.image_processor(
            images=[row["image"] for row in examples],
            max_num_patches=self.max_patches, return_tensors="pt",
        )
        length = max(len(row["input_ids"]) for row in examples)
        ids = torch.full((len(examples), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(examples):
            size = len(row["input_ids"])
            ids[index, :size] = torch.tensor(row["input_ids"])
            labels[index, :size] = torch.tensor(row["labels"])
            attention[index, :size] = 1
        return dict(images, input_ids=ids, labels=labels, attention_mask=attention)
