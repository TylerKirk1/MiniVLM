"""Load the fixed backbones and save only the learned bridge/LoRA weights."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Siglip2ImageProcessor, Siglip2VisionModel,
)

from .model import MiniVLM
from .models import BridgeConfig


TEXT_MODEL = "Qwen/Qwen3-1.7B"
VISION_MODEL = "google/siglip2-so400m-patch16-naflex"


@dataclass
class Settings:
    num_latents: int = 256
    max_patches: int = 256
    max_text_tokens: int = 512
    text_revision: str = "main"
    vision_revision: str = "main"

    def __post_init__(self):
        if min(self.num_latents, self.max_patches, self.max_text_tokens) < 1:
            raise ValueError("Token and patch limits must be positive")


def load_processors(settings):
    # Resolve moving refs before loading any tokenizer, processor, or weights.
    text_config = AutoConfig.from_pretrained(TEXT_MODEL, revision=settings.text_revision)
    vision_config = AutoConfig.from_pretrained(VISION_MODEL, revision=settings.vision_revision)
    settings.text_revision = text_config._commit_hash
    settings.vision_revision = vision_config._commit_hash
    if settings.num_latents + settings.max_text_tokens > text_config.max_position_embeddings:
        raise ValueError("Visual and text token budgets exceed the language context limit")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=settings.text_revision)
    processor = Siglip2ImageProcessor.from_pretrained(
        VISION_MODEL, revision=settings.vision_revision
    )
    return tokenizer, processor


def require_training_device():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA-enabled PyTorch and a BF16-capable NVIDIA GPU")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install the quantization extra: pip install -e ".[train]"') from exc


def add_lora(language):
    return get_peft_model(language, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))


def load_model(settings, stage="alignment", checkpoint=None, training=False):
    require_training_device()
    if stage not in {"alignment", "qlora"}:
        raise ValueError("Stage must be alignment or qlora")
    language = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL, revision=settings.text_revision, dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    language = prepare_model_for_kbit_training(
        language, use_gradient_checkpointing=training,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    language.config.use_cache = False
    metadata = read_checkpoint(checkpoint) if checkpoint is not None else None
    if metadata is not None and metadata["stage"] == "qlora":
        language = PeftModel.from_pretrained(language, Path(checkpoint) / "adapter",
                                             is_trainable=training)
    elif stage == "qlora":
        language = add_lora(language)
    vision = Siglip2VisionModel.from_pretrained(
        VISION_MODEL, revision=settings.vision_revision, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    bridge_config = BridgeConfig(
        vision_feature_dim=vision.config.hidden_size, latent_dim=512,
        num_latents=settings.num_latents, depth=2, num_heads=8, mlp_ratio=4,
        projector_hidden_dim=language.config.hidden_size,
        llm_hidden_size=language.config.hidden_size,
    ) if metadata is None else BridgeConfig(**metadata["bridge"])
    model = MiniVLM(vision, language, bridge_config)
    model.bridge.to("cuda")
    if checkpoint is not None:
        model.bridge.load_state_dict(load_file(str(Path(checkpoint) / "bridge.safetensors")))
    model.train(training)
    return model


def save_checkpoint(model, settings, stage, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_file({key: value.detach().cpu().contiguous()
               for key, value in model.bridge.state_dict().items()},
              str(output / "bridge.safetensors"))
    if stage == "qlora":
        model.language.save_pretrained(output / "adapter", save_embedding_layers=False)
    metadata = {"format_version": 1, "stage": stage, "settings": asdict(settings),
                "bridge": asdict(model.bridge.config)}
    # Write metadata last: an interrupted save must not look like a complete checkpoint.
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def read_checkpoint(path):
    path = Path(path)
    metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("stage") not in {"alignment", "qlora"}:
        raise ValueError("Unsupported MiniVLM checkpoint")
    if not (path / "bridge.safetensors").is_file():
        raise ValueError("Checkpoint is missing bridge weights")
    if metadata["stage"] == "qlora" and not (path / "adapter" / "adapter_model.safetensors").is_file():
        raise ValueError("Checkpoint is missing LoRA weights")
    Settings(**metadata["settings"])
    BridgeConfig(**metadata["bridge"])
    return metadata
