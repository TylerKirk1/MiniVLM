# Datasets And Training

## Training Objective

The first version should optimize for practical capability on:
- object identification
- short-form OCR
- basic visual question answering

The training setup should not chase broad benchmark coverage before those three tasks work acceptably.

## Dataset Buckets

The working data plan is a mixture of four buckets.

### 1. Image-Text Alignment Data

Purpose:
- teach the connector and projector to map visual features into a usable language space

Examples:
- image-caption pairs
- short descriptive image-text pairs
- general-purpose image instruction data with simple answers

### 2. OCR-Focused Data

Purpose:
- force the model to attend to text regions
- prevent the system from becoming caption-only

Examples:
- scene text QA
- screenshot text QA
- image-to-transcription tasks
- synthetic text-rendered images if needed

### 3. Object-Centric Data

Purpose:
- strengthen coarse and mid-grained recognition
- improve answers to "what is this?" and "what objects are present?" prompts

Examples:
- object question answering
- object-rich captions
- category identification prompts derived from labeled datasets

### 4. General Multimodal SFT Data

Purpose:
- teach instruction following once the image pathway works

Examples:
- conversational VQA
- caption-follow-up question pairs
- short grounded reasoning prompts

## Recommended Mixture

Initial starting point:
- `30%` image-text alignment
- `30%` OCR-focused
- `25%` object-centric
- `15%` general multimodal SFT

Why this is intentionally skewed:
- object identification is easier to recover from generic data
- OCR usually underperforms unless it is directly represented in the training mix

This ratio should be revisited after the first evaluation pass.

## Data Formatting

All data should be converted into a unified chat or instruction format.

Example shape:

```text
User: <image> What text is shown on the sign?
Assistant: MAIN ST
```

or:

```text
User: <image> Identify the main object in the image.
Assistant: red fire hydrant
```

Formatting requirements:
- keep answers short when the task demands exactness
- avoid verbose OCR targets that encourage paraphrasing
- normalize casing policy intentionally during evaluation
- store metadata for task type so mixtures can be controlled later

## Training Phases

### Phase 1: Connector Alignment

Train:
- resampler
- projector

Freeze:
- vision encoder
- LLM

Possible data focus:
- image-caption pairs
- simple OCR prompts with short targets
- basic object labels in natural language form

Success condition:
- the model clearly conditions on the image rather than answering blindly from text priors

### Phase 2: Multimodal SFT

Train:
- resampler
- projector
- `QLoRA` adapters on the LLM

Freeze:
- most or all of the vision encoder

Data focus:
- mixed OCR, object-ID, and VQA data
- task-balanced batches if possible

Success condition:
- measurable gains on local OCR and object-ID evals

### Phase 3: Optional Vision Refinement

Train:
- small upper subset of the vision encoder
- resampler
- projector
- `QLoRA` adapters

Only run this phase if:
- OCR stays weak despite targeted data
- object recognition remains brittle on visually similar items
- evidence suggests the bottleneck is visual, not linguistic

## Initial Hyperparameter Direction

These are starting points, not fixed commitments.

### Image Side

- start with one or two supported image-size buckets rather than many
- preserve enough resolution for text to remain legible
- avoid introducing variable-resolution complexity before the base training loop works

### Sequence Side

- start with a modest text length budget
- account for the `256` visual tokens as part of the effective context budget
- prefer stable, short targets early in training

### Optimization

- use `AdamW`
- use gradient accumulation aggressively
- start with conservative learning rates for the LLM adapters
- allow a higher learning rate for the resampler/projector than for the LoRA layers

### LoRA / QLoRA

Likely first tuning knobs:
- rank
- alpha
- target module coverage
- dropout

The initial implementation should expose these cleanly in config rather than hard-coding them.

## Data Risks

- OCR labels may be noisy or inconsistently normalized
- caption-heavy datasets can make the model fluent but visually shallow
- object labels from detection data often need prompt conversion and cleanup
- synthetic text data may help OCR but can distort the target distribution if overused

## Recommended First Eval-Oriented Training Slice

Before building a large training corpus, assemble a smaller but task-balanced slice that includes:
- clean object-ID examples
- clean OCR examples from the intended domain
- simple VQA examples

This slice should be small enough to train quickly and good enough to expose whether the architecture is fundamentally working.
