# VLM

Early-stage vision-language model project built around a small text backbone and a pretrained vision encoder.

The current working plan is:
- Text backbone: `Qwen3-1.7B`
- Vision encoder: `SigLIP-2 NaFlex So400M`
- Visual adapter: learned `256`-token resampler
- Text adaptation: `QLoRA` on the LLM
- Primary baseline: `Qwen3.5-2B`

## Project Goal

Build a compact multimodal model that is reasonably capable at:
- general object identification
- short-form OCR in natural images and screenshots
- simple image question answering

This project is not currently targeting:
- dense document OCR
- long-form document understanding
- fine-grained grounding or detection
- state-of-the-art benchmark performance

## Hardware Assumption

Initial development assumes a home workstation with an `RTX 4070 12GB`.

That constraint drives several design choices:
- use a pretrained vision encoder instead of training one from scratch
- compress image features before feeding them to the LLM
- use `QLoRA` instead of full LLM finetuning
- keep early experiments narrow and measurable

## Current Architecture Direction

High-level flow:

1. Encode the image with `SigLIP-2 NaFlex So400M`
2. Convert vision features into a fixed learned set of `256` visual tokens
3. Project visual tokens into the LLM hidden space
4. Concatenate visual tokens with the text prompt
5. Train the multimodal stack with a frozen or mostly frozen vision tower and `QLoRA` on the LLM

See [docs/architecture.md](/home/tylerkirk/projects/VLM/docs/architecture.md) for the detailed design.

## Training Strategy

Planned training phases:

1. Connector alignment
   Train the visual resampler and projector first, with the LLM and vision encoder mostly frozen.
2. Multimodal SFT
   Apply `QLoRA` to the LLM and continue training on OCR, object-ID, captioning, and VQA-style data.
3. Optional vision refinement
   Only if needed, unfreeze a small portion of the upper vision stack for targeted refinement.

See [docs/datasets-and-training.md](/home/tylerkirk/projects/VLM/docs/datasets-and-training.md) for the working recipe.

## Evaluation Direction

The initial baseline plan is to compare against `Qwen3.5-2B` on a focused local eval suite covering:
- OCR
- object identification
- simple visual QA
- runtime and memory behavior

See [docs/evaluation.md](/home/tylerkirk/projects/VLM/docs/evaluation.md) for the proposed comparison setup.

## Planned Repo Layout

This is the intended structure as implementation begins:

```text
.
|-- README.md
|-- docs/
|   |-- architecture.md
|   |-- datasets-and-training.md
|   `-- evaluation.md
|-- configs/
|-- scripts/
|-- src/
`-- experiments/
```

Only the documentation exists right now. Code layout can be adjusted once the training stack is chosen.

## Near-Term Milestones

1. Finalize architecture details and training interfaces
2. Scaffold dataset ingestion and prompt formatting
3. Implement the resampler/projector path
4. Run a memory-fit smoke test on the `4070`
5. Train a small alignment run
6. Run the first local comparison against `Qwen3.5-2B`

## Open Decisions

- whether to keep `NaFlex` dynamic-resolution behavior in the first implementation or temporarily pin input sizes
- whether `256` visual tokens is sustainable on `12GB` once prompt lengths and batch sizes are realistic
- whether the first OCR objective should focus on screenshots, scene text, or mixed text domains
- how much custom local evaluation data to collect before training

## Status

The repo is currently in planning mode. The documents in `docs/` are the source of truth for the first implementation pass.
