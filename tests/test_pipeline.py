"""Offline checks using tiny real SigLIP/Qwen models; no optimizer updates."""

from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM,
                          Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel)

from vlm.data import ImageQACollator, ImageQADataset, encode_example, prompt_ids
from vlm.model import MiniVLM
from vlm.models import BridgeConfig
from vlm.runtime import Settings, add_lora, read_checkpoint, save_checkpoint
from vlm.train import main


def tiny_tokenizer():
    words = ["<pad>", "<|im_end|>", "<unk>", "<|im_start|>", "<think>", "</think>",
             "user", "assistant", "What", "color?", "red", "blue", "Read", "this", "STOP"]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<|im_end|>", unk_token="<unk>",
        additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] "
        "+ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"
    )
    return tokenizer


def tiny_model(lora=False):
    vision = Siglip2VisionModel(Siglip2VisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, patch_size=4, num_patches=16,
    ))
    language = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
        attention_dropout=0.0,
    )).requires_grad_(False)
    if lora:
        language = add_lora(language)
    return MiniVLM(vision, language, BridgeConfig(16, 16, 4, 1, 4, 2, 32, 32))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer = tiny_tokenizer()
        self.processor = Siglip2ImageProcessor(patch_size=4, max_num_patches=16)
        self.collator = ImageQACollator(self.tokenizer, self.processor, 16)
        Image.new("RGB", (19, 11), "red").save(self.root / "image.png")
        self.manifest = self.root / "train.jsonl"
        self.write_rows([{"image": "image.png", "question": "What color?", "answer": "red"},
                         {"image": "image.png", "question": "Read", "answer": "STOP"}])

    def write_rows(self, rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def batch(self):
        dataset = ImageQADataset(self.manifest, self.tokenizer, 64)
        return self.collator([dataset[0], dataset[1]])

    def test_answer_only_labels_and_right_padding(self):
        batch = self.batch()
        prompt = prompt_ids(self.tokenizer, "What color?")
        self.assertTrue((batch["labels"][0, :len(prompt)] == -100).all())
        self.assertEqual(batch["labels"][0, -1].item(), self.tokenizer.eos_token_id)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)
        self.assertIn("spatial_shapes", batch)
        self.assertTrue((batch["pixel_attention_mask"] == 0).any())

    def test_rejects_truncation_bad_schema_and_missing_images(self):
        with self.assertRaisesRegex(ValueError, "Shorten"):
            encode_example(self.tokenizer, "What color?", "red", 2)
        for row in [{"image": "missing.png", "question": "Read", "answer": "STOP"},
                    {"image": "image.png", "question": "Read", "answer": ""},
                    {"image": "image.png", "question": "Read", "answer": "STOP", "typo": 1}]:
            self.write_rows([row])
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "train.jsonl:1"):
                ImageQADataset(self.manifest, self.tokenizer)

    def test_empty_and_corrupt_data_fail_before_model_loading(self):
        self.manifest.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No examples"):
            ImageQADataset(self.manifest, self.tokenizer)
        self.write_rows([{"image": "image.png", "question": "Read", "answer": "STOP"}])
        (self.root / "image.png").write_text("not an image", encoding="utf-8")
        with patch("vlm.train.load_processors", return_value=(self.tokenizer, self.processor)), \
             patch("vlm.train.load_model") as load, \
             self.assertRaisesRegex(ValueError, "Cannot decode"):
            main([str(self.manifest)])
        load.assert_not_called()

    def test_real_image_to_language_loss_and_gradients_without_weight_updates(self):
        model = tiny_model().train()
        model.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        batch = self.batch()
        result = model(**batch)
        labels = torch.cat((torch.full((2, 4), -100), batch["labels"]), dim=1)
        expected = torch.nn.functional.cross_entropy(
            result.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        torch.testing.assert_close(result.loss, expected)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertFalse(model.vision.training)
        self.assertTrue(all(p.grad is None for p in model.vision.parameters()))
        self.assertTrue(all(p.grad is None for p in model.language.parameters()))
        for parameter in model.bridge.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.bridge.resampler.latents.grad.abs().sum().item(), 0)

    def test_padding_is_ignored_and_images_condition_language(self):
        model = tiny_model().eval()
        batch = self.batch()
        changed = {key: value.clone() for key, value in batch.items()}
        padding = changed["pixel_attention_mask"] == 0
        changed["pixel_values"][padding] = 1000
        with torch.no_grad():
            original = model(**batch).logits
            torch.testing.assert_close(original, model(**changed).logits, atol=1e-5, rtol=1e-4)
            changed["pixel_values"] = -batch["pixel_values"]
            self.assertFalse(torch.allclose(original, model(**changed).logits))

    def test_lora_gradients_and_generation(self):
        model = tiny_model(lora=True).train()
        model(**self.batch()).loss.backward()
        trained = [(name, p) for name, p in model.language.named_parameters() if p.requires_grad]
        self.assertTrue(trained)
        self.assertTrue(all("lora_" in name for name, _ in trained))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trained))
        ids = torch.tensor([prompt_ids(self.tokenizer, "What color?")])
        inputs = self.processor(images=[Image.new("RGB", (19, 11), "red")], return_tensors="pt")
        generated = model.generate(**inputs, input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=3)
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], 0)
        self.assertLessEqual(generated.shape[1], 3)  # Only new tokens, no prompt to strip.

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        from safetensors.torch import load_file
        from peft import PeftModel
        for stage in ("alignment", "qlora"):
            model = tiny_model(lora=stage == "qlora")
            output = self.root / stage
            settings = Settings(num_latents=4, max_patches=16, max_text_tokens=64)
            save_checkpoint(model, settings, stage, output)
            metadata = read_checkpoint(output)
            self.assertEqual(metadata["settings"], asdict(settings))
            restored = tiny_model()
            restored.bridge.load_state_dict(load_file(str(output / "bridge.safetensors")))
            for name, value in model.bridge.state_dict().items():
                torch.testing.assert_close(value, restored.bridge.state_dict()[name])
            if stage == "qlora":
                restored.language = PeftModel.from_pretrained(restored.language, output / "adapter")
                for name, value in model.language.state_dict().items():
                    if "lora_" in name:
                        torch.testing.assert_close(value, restored.language.state_dict()[name])
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, settings, stage, output)

    def test_preflight_never_loads_weights_or_writes_output(self):
        output = self.root / "output"
        with patch("vlm.train.load_processors", return_value=(self.tokenizer, self.processor)), \
             patch("vlm.train.load_model") as load, patch("vlm.train.train_steps") as train, \
             patch("vlm.train.save_checkpoint") as save, \
             patch("vlm.train.require_training_device", side_effect=RuntimeError("test CPU")):
            main([str(self.manifest), "--output", str(output)])
        load.assert_not_called()
        train.assert_not_called()
        save.assert_not_called()
        self.assertFalse(output.exists())

    def test_training_requires_explicit_complete_arguments(self):
        for flags in (["--train"], ["--stage", "qlora"], ["--steps", "0"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), \
                 patch("vlm.train.load_processors") as processors, \
                 patch("vlm.train.load_model") as load, \
                 self.assertRaises(SystemExit) as error:
                main([str(self.manifest), *flags])
            self.assertEqual(error.exception.code, 2)
            processors.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
