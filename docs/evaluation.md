# Evaluation

## Purpose

The first evaluation plan is designed to answer three practical questions:

1. Does the model actually use the image?
2. Is it acceptable at object identification?
3. Is it at least somewhat useful for short-form OCR?

The main external comparison point is `Qwen3.5-2B`.

## Baseline Strategy

Use `Qwen3.5-2B` as a reference system for:
- qualitative answer quality
- OCR behavior
- object naming accuracy
- latency and memory tradeoffs, where applicable

This is a pragmatic baseline, not a claim of architectural equivalence.

## Evaluation Tracks

### OCR

Primary target:
- short visible text in screenshots, signs, labels, and UI elements

Measure:
- exact match on short strings
- normalized exact match if casing or punctuation is not semantically important

Failure patterns to track:
- hallucinated text
- partial reads
- substitutions between visually similar characters
- answering the question without reading the image text

### Object Identification

Primary target:
- identify the main object or list major visible objects

Measure:
- exact or normalized string match for single-object prompts
- category overlap for multi-object prompts

Failure patterns to track:
- over-generic labels
- visually adjacent object confusion
- bias toward common objects even when wrong

### General Visual QA

Primary target:
- basic factual questions grounded in a single image

Measure:
- short-answer accuracy on a small curated set

Failure patterns to track:
- language-prior answers
- weak grounding
- loss of answer precision once prompts become slightly conversational

## Local Eval Set

The project should maintain a small local evaluation set that reflects actual intended use.

Recommended composition:
- screenshots with visible UI text
- natural images with signs or labels
- common household or street objects
- a small number of visually confusing pairs

This local set is more valuable early on than broad benchmark chasing.

## Minimum Success Criteria For MVP

The first implementation should be considered viable only if it can:
- outperform text-only guessing on image-conditioned prompts
- correctly identify common objects in straightforward images
- read short, clean text in a meaningful portion of OCR examples

If it fails any of those, architecture and data debugging should come before larger training runs.

## Comparison Rules

When comparing against `Qwen3.5-2B`:
- use the same prompts whenever possible
- keep answer formatting constraints explicit
- separate qualitative comparisons from exact-match comparisons
- log obvious failure modes, not just aggregate scores

## What To Measure Besides Accuracy

- VRAM usage during train and eval
- effective image resolution used
- visual token count
- prompt length sensitivity
- inference latency on the local machine

These metrics matter because the project target is a usable small VLM, not just a one-off training run.

## Evaluation Order

1. sanity-check prompts with a few handpicked images
2. run the local eval set on the untrained or minimally adapted system
3. rerun after connector alignment
4. rerun after multimodal SFT
5. compare against `Qwen3.5-2B`

That sequence should make regressions easier to detect than running only a final benchmark pass.
