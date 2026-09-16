# MiniVLM

A small vision-language model experiment for object recognition, short-form OCR,
and simple image questions, intended for an RTX 4070 with 12 GB of VRAM.

**Status: standalone building blocks only.** There is no image encoder or language
model integration, dataset loader, training loop, inference command, or trained
checkpoint yet. The smoke check uses random features, not images.

## What works

- A learned cross-attention resampler compresses vision features into a fixed
  number of visual tokens.
- An MLP projector maps those tokens to a language model's embedding dimension.
- Text metrics provide exact match, normalized exact match, and character error
  rate (CER).

```text
vision features [batch, patches, vision_dim]
  -> learned resampler
  -> MLP projector
  -> visual tokens [batch, num_latents, llm_hidden_size]
```

The bridge accepts feature tensors with no padding. It does not handle images,
patch masks, text prompts, or language generation. Its weights start randomly
initialized and must be trained before they can carry useful visual information.

## Run locally

Python 3.11 or newer is required. From the repository root, in an activated
virtual environment:

```sh
python -m pip install -e .
python scripts/check_shapes.py
python -m unittest discover -s tests -v
```

PyTorch is the only runtime dependency. The smoke check runs on CPU without
downloading model weights. It checks a forward and backward pass with the original
provisional bridge dimensions: 1,152 input features, 256 visual tokens, and 2,048
output features. Expected shapes with the default arguments:

```text
input shape:  (1, 64, 1152)
output shape: (1, 256, 2048)
backward pass: finite gradients
```

These dimensions are a synthetic test preset, not validated checkpoint metadata
or evidence that the complete model fits in 12 GB. Use `--batch-size` and
`--source-tokens` to vary the input. Construct a `BridgeConfig` in Python for other
bridge dimensions.

## Code

- `src/vlm/models/`: `BridgeConfig`, `VisionLanguageBridge`, resampler, and projector.
- `src/vlm/eval/metrics.py`: standalone text-scoring functions.
- `scripts/check_shapes.py`: CPU smoke check.
- `tests/`: bridge behavior and metric checks, using Python's `unittest`.

The metrics operate on individual strings, not datasets. Normalized exact match
ignores case and repeated whitespace by default; punctuation stripping is opt-in.
CER uses raw characters and can exceed 1. For an empty target, it returns 0 for an
empty prediction and 1 otherwise.

## Intended next step

The original design pairs a frozen SigLIP-2 NaFlex So400M vision encoder with
Qwen3-1.7B through a 256-token bridge. The proposed training sequence is connector
alignment first, then QLoRA adaptation of the language model, with Qwen3.5-2B as a
comparison target. None of that integration has been implemented or validated.

The next useful milestone is one real image-to-answer forward and backward pass:
verify checkpoint dimensions, handle vision padding, connect visual and text
embeddings with correct attention and loss masks, and measure memory use. Larger
dataset recipes and training schedules should wait until that path works.
