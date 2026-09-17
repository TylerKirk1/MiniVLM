# MiniVLM

A compact vision-language experiment for **object identification, short OCR, and
simple image questions** on an RTX 4070 (12 GB). One image, one question, one short
answer per example.

```text
image -> frozen SigLIP-2 NaFlex -> 256-token bridge -> Qwen3-1.7B -> answer
```

The pipeline is implemented; no model has been trained. An untrained bridge will
not produce useful visual answers. Full-size training memory use and answer
quality have not been measured.

## Setup

Python 3.11+; training and checkpoint inference require an NVIDIA GPU with BF16
support. On Windows PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[train]"
```

The CUDA wheel above matches the tested RTX 4070 setup; see
[PyTorch installation options](https://pytorch.org/get-started/previous-versions/)
for other systems. CPU-only development can use `python -m pip install -e .`.
Transformers and PEFT versions are pinned to the tested APIs. No torchvision,
dataset framework, experiment tracker, or logging service is needed.

## Data and preflight

Supply a UTF-8 JSONL file such as `data/train.jsonl`. Each line has exactly these
three nonempty string fields:

```json
{"image":"images/sign.png","question":"What text is on the sign?","answer":"MAIN ST"}
```

Image paths are relative to the manifest. Use real images and accurate, concise
answers. Keep evaluation images separate from training data. Local data is
ignored by Git; no sample dataset or placeholder images are shipped.

```sh
python -m vlm.train data/train.jsonl
```

This validates every record, decodes and preprocesses every image, checks text
lengths, and reports whether CUDA and quantization dependencies are available.
It downloads only tokenizer/processor metadata into the normal Hugging Face
cache. **It does not load pretrained weights, train, or create run files.**

Defaults are 256 visual tokens, at most 256 image patches, and 512 text tokens.
Use `--visual-tokens`, `--max-patches`, and `--max-text-tokens` if needed.
Overlong examples are rejected rather than silently truncating OCR answers.

## Training, when ready

These commands **do update weights**. They are documented for a future run.

First align the bridge while both backbones stay frozen:

```sh
python -m vlm.train data/train.jsonl --train --steps 100 --output checkpoints/alignment
```

Then initialize from that bridge and adapt Qwen with QLoRA:

```sh
python -m vlm.train data/train.jsonl --stage qlora --init checkpoints/alignment --train --steps 100 --output checkpoints/qlora
```

The example step counts are smoke-run sizes, not a validated training recipe.
Each optimizer update averages 16 single-example microbatches by default
(`--accumulation`). Data reshuffles and repeats until `--steps` is reached.
The bridge uses AdamW at 2e-4, LoRA at 5e-5, with gradient clipping.

Both stages use a 4-bit NF4 language backbone, BF16 computation, gradient
checkpointing, and a frozen vision encoder. Only answer tokens and the answer's
EOS contribute to the loss. Vision padding, prompt tokens, and text padding are
masked. Training and inference share Qwen's non-thinking chat prompt.

Training prints brief progress to the terminal and saves **one final checkpoint**:
bridge weights, backbone revisions/settings, and LoRA weights when applicable.
There are no log files, dashboards, periodic checkpoint piles, or duplicated
backbone weights. Existing outputs are never overwritten. Interrupted runs are
not resumable; choose a new output directory to restart.

## Use a trained checkpoint

```sh
python -m vlm.predict checkpoints/qlora path/to/image.png "What text is on the sign?"
```

This loads the original backbone revisions plus the saved bridge/adapter and
prints a greedy answer. It downloads pretrained weights if they are not cached.
Use `--max-new-tokens` to change the 64-token answer limit.

## Verification and code

```sh
python -m unittest discover -s tests -v
```

Tests use temporary images and tiny, randomly initialized **real SigLIP and Qwen
models**, not downloaded checkpoints. Forward/backward checks verify integration
without optimizer updates. They cover image masking, answer-only loss, gradients,
generation, checkpoint round-trips, and the preflight guard. A recording
optimizer checks accumulation arithmetic without modifying weights.
When CUDA and the training extra are installed, the suite also exercises actual
4-bit loading, both stages, and checkpoint inference on tiny local models.

- `src/vlm/data.py`: JSONL validation, image loading, tokenization, batching.
- `src/vlm/model.py`, `src/vlm/models/`: backbone integration and visual bridge.
- `src/vlm/runtime.py`: pretrained loading, LoRA, compact checkpoints.
- `src/vlm/train.py`, `src/vlm/predict.py`: the two commands.
- `src/vlm/eval/metrics.py`: exact match, normalized match, character error rate.

The architecture follows the published
[Qwen3-1.7B configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json),
[SigLIP-2 NaFlex interface](https://huggingface.co/docs/transformers/model_doc/siglip2),
and [PEFT quantization workflow](https://huggingface.co/docs/peft/developer_guides/quantization).
Model dimensions come from the loaded checkpoints. Broader benchmarks,
multi-image/chat support, dataset downloading, and vision finetuning are outside
this project's current scope.
