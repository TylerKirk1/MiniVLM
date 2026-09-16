"""Optional real 4-bit integration check; temporary random weights, no training."""

import gc
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from transformers import Siglip2Config, Siglip2Model, Siglip2TextConfig

from vlm.data import encode_example, ImageQACollator
from vlm.runtime import Settings, load_model, save_checkpoint
from test_pipeline import tiny_model, tiny_tokenizer


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes"),
                     "CUDA-enabled PyTorch and the train extra are required")
class QuantizedPipelineTests(unittest.TestCase):
    def test_actual_quantized_loader_both_stages_and_checkpoint_inference(self):
        from transformers import Siglip2ImageProcessor
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = tiny_model()
            base.language.save_pretrained(root / "language")
            # The published SigLIP checkpoint includes a text tower. Verify extraction
            # of only its vision tower through the actual from_pretrained path.
            full_vision = Siglip2Model(Siglip2Config(
                vision_config=base.vision.config.to_dict(),
                text_config=Siglip2TextConfig(vocab_size=32, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=4).to_dict(),
            ))
            full_vision.save_pretrained(root / "vision")
            expected_vision = full_vision.vision_model.embeddings.patch_embedding.weight.clone()
            del base, full_vision
            settings = Settings(num_latents=4, max_patches=16, max_text_tokens=64)
            tokenizer = tiny_tokenizer()
            processor = Siglip2ImageProcessor(patch_size=4, max_num_patches=16)
            ids, labels = encode_example(tokenizer, "What color?", "red", 64)
            collator = ImageQACollator(tokenizer, processor, 16)
            batch = collator([{"image": Image.new("RGB", (19, 11), "red"),
                               "input_ids": ids, "labels": labels}])
            batch = {key: value.cuda() for key, value in batch.items()}
            with patch("vlm.runtime.TEXT_MODEL", str(root / "language")), \
                 patch("vlm.runtime.VISION_MODEL", str(root / "vision")):
                checkpoint = None
                for stage in ("alignment", "qlora"):
                    model = load_model(settings, stage, checkpoint, training=True)
                    self.assertTrue(model.language.is_loaded_in_4bit)
                    torch.testing.assert_close(
                        model.vision.vision_model.embeddings.patch_embedding.weight.cpu(),
                        expected_vision.to(torch.bfloat16),
                    )
                    before = model.bridge.resampler.latents.detach().clone()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model(**batch)
                    self.assertTrue(torch.isfinite(result.loss))
                    result.loss.backward()
                    gradient = model.bridge.resampler.latents.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None for p in model.vision.parameters()))
                    if stage == "alignment":
                        self.assertTrue(all(p.grad is None for p in model.language.parameters()))
                    else:
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.language.parameters() if p.requires_grad))
                    torch.testing.assert_close(before, model.bridge.resampler.latents)
                    checkpoint = root / stage
                    save_checkpoint(model, settings, stage, checkpoint)
                    model.eval()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        expected_logits = model(**batch).logits.cpu()
                    del result, model
                    gc.collect()
                    torch.cuda.empty_cache()
                restored = load_model(settings, "qlora", checkpoint)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.testing.assert_close(restored(**batch).logits.cpu(), expected_logits)
                    inputs = {key: value for key, value in batch.items() if key != "labels"}
                    # Remove the answer from the synthetic prompt for generation.
                    inputs["input_ids"] = inputs["input_ids"][:, :-2]
                    inputs["attention_mask"] = inputs["attention_mask"][:, :-2]
                    generated = restored.generate(**inputs, max_new_tokens=3)
                self.assertGreater(generated.shape[1], 0)
                self.assertLessEqual(generated.shape[1], 3)
                del restored
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
