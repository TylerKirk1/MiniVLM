import unittest

import torch

from vlm.models import BridgeConfig, VisionLanguageBridge


class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.bridge = VisionLanguageBridge(BridgeConfig(
            vision_feature_dim=12,
            latent_dim=16,
            num_latents=4,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            projector_hidden_dim=24,
            llm_hidden_size=20,
        ))

    def test_variable_patch_counts_produce_fixed_token_budget(self):
        for patches in (1, 7, 19):
            with self.subTest(patches=patches), torch.no_grad():
                output = self.bridge(torch.randn(2, patches, 12))
                self.assertEqual(tuple(output.shape), (2, 4, 20))
                self.assertTrue(torch.isfinite(output).all().item())

    def test_gradients_reach_source_and_every_trainable_parameter(self):
        source = torch.randn(2, 7, 12, requires_grad=True)
        output = self.bridge(source)
        torch.nn.functional.mse_loss(output, torch.randn_like(output)).backward()
        gradients = {name: p.grad for name, p in self.bridge.named_parameters()}
        gradients["source"] = source.grad
        for name, gradient in gradients.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_predictions_depend_on_features_without_mixing_batch_items(self):
        source = torch.randn(2, 7, 12)
        changed = source.clone()
        changed[1] = torch.randn(7, 12)
        with torch.no_grad():
            original = self.bridge(source)
            modified = self.bridge(changed)
            single = self.bridge(source[:1])
        torch.testing.assert_close(original[0], modified[0])
        torch.testing.assert_close(original[:1], single)
        self.assertFalse(torch.allclose(original[1], modified[1]))

    def test_rejects_features_without_batch_dimension(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.bridge(torch.randn(7, 12))

    def test_rejects_empty_or_misaligned_patch_masks(self):
        source = torch.randn(2, 7, 12)
        for mask in (torch.zeros(2, 7), torch.ones(2, 6)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                self.bridge(source, mask)

    def test_padded_patches_cannot_change_visual_tokens(self):
        source = torch.randn(2, 7, 12)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[:, -2:] = False
        changed = source.clone()
        changed[:, -2:] = 1000
        with torch.no_grad():
            torch.testing.assert_close(self.bridge(source, mask), self.bridge(changed, mask))


if __name__ == "__main__":
    unittest.main()
