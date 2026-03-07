# Architecture

## Working Configuration

- LLM: `Qwen3-1.7B`
- Vision encoder: `SigLIP-2 NaFlex So400M`
- Visual interface: learned `256`-token resampler
- Language adaptation: `QLoRA`
- Main comparison target: `Qwen3.5-2B`

This document describes the intended first architecture, not a finalized implementation.

## System Outline

The model will follow a standard adapter-style VLM design:

```text
image
  -> vision encoder
  -> patch/grid features
  -> learned resampler
  -> projector to LLM hidden size
  -> visual tokens

visual tokens + text prompt
  -> Qwen3-1.7B
  -> next-token prediction / instruction response
```

## Component Decisions

### Text Backbone

`Qwen3-1.7B` is the current choice because it is small enough to be practical on a `4070 12GB` while still leaving room for multimodal context and `QLoRA`.

Reasons to prefer this size class:
- lower VRAM pressure during multimodal training
- faster iteration for architecture/debugging work
- less risk that the connector work is blocked by hardware limits

### Vision Encoder

`SigLIP-2 NaFlex So400M` provides a strong pretrained visual backbone with better headroom than older lightweight CLIP variants.

Key assumptions:
- the encoder starts pretrained and frozen
- most value comes from reuse, not retraining from scratch
- dynamic or flexible resolution is useful for OCR, but may need to be simplified in the first pass if tooling support is weak

### Learned Resampler

The vision encoder will likely emit more spatial features than the `4070` budget can comfortably pass into the LLM directly.

The resampler exists to:
- compress variable-size vision features into a fixed token budget
- preserve more detail than coarse pooling
- make multimodal sequence length predictable

Current target: `256` learned visual tokens.

That number should be treated as a starting point, not a guarantee. If VRAM or training speed is unacceptable, the first fallback should be reducing the visual token count before changing the backbone.

### Projector

After resampling, visual tokens must be projected into the LLM hidden space.

Expected first implementation:
- a small MLP projector
- LayerNorm on the visual side if needed
- optional learned image boundary tokens depending on prompt format

The projector should stay simple until there is evidence it is the bottleneck.

### QLoRA on the LLM

The initial adaptation path is `QLoRA` on the text backbone rather than full finetuning.

Why:
- it fits the hardware envelope better
- it keeps iteration speed reasonable
- it limits the number of moving parts while the vision-text interface is still stabilizing

Expected target modules will include the main attention and MLP projections, but exact coverage should be verified once the model loading path is chosen.

## Sequence Construction

The intended prompt structure is conceptually:

```text
<vision_tokens> + user prompt + assistant target
```

Open implementation questions:
- whether to use explicit placeholder tokens in the tokenizer stream
- whether image features should be inserted at the front of the sequence or wrapped by learned delimiters
- how many text tokens can be preserved once `256` visual tokens are present

The first implementation should prefer the simplest sequence layout that works reliably.

## Training Phases

### Phase 1: Connector Alignment

Objective:
- train the resampler and projector
- keep the vision encoder frozen
- keep the LLM frozen or nearly frozen

Goal:
- prove that image features can condition the language model at all

### Phase 2: Multimodal SFT

Objective:
- enable `QLoRA` on the LLM
- continue training the visual interface
- train on instruction-style multimodal data

Goal:
- improve answer quality on OCR, object identification, and simple VQA

### Phase 3: Optional Vision Refinement

Objective:
- selectively unfreeze a small upper portion of the vision stack
- run a short targeted refinement phase

Goal:
- recover quality if OCR or fine-grained visual discrimination remains weak after phase 2

This phase should be skipped unless there is a clear failure mode that the connector and LLM adaptation cannot fix.

## Constraints on a 4070 12GB

The main architecture risks under the current hardware budget are:
- too many visual tokens
- too much text context
- overly large image inputs
- training instability once quantization, LoRA, and multimodal batching interact

Practical implications:
- start with low batch sizes and gradient accumulation
- keep the first text context window modest
- verify memory fit before scaling data or image size
- treat `256` visual tokens as an explicit experiment

## First Fallbacks if Memory Is Tight

In order of preference:

1. reduce visual token count
2. reduce text sequence length
3. reduce image resolution or bucket sizes
4. narrow LoRA target coverage
5. move to a smaller or simpler vision feature path

Changing the core backbone should be a later decision, not the first reaction.

## Open Questions

- Is `256` the right resampler budget, or should the initial implementation start at `128` and scale upward?
- Should the first version use fixed-size image buckets even if the long-term plan is `NaFlex`?
- Is OCR quality better served by more image resolution, more visual tokens, or a more OCR-heavy data mix?
- Does the chosen prompt format need explicit image sentinel tokens for stable training?
