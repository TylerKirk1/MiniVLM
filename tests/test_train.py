"""Check accumulation/control flow with a recording optimizer that never updates weights."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import torch

from vlm.train import train_steps


class RecordingOptimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.gradients = []

    def zero_grad(self, set_to_none=True):
        self.parameter.grad = None

    def step(self):
        self.gradients.append(self.parameter.grad.item())


class SyntheticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, coefficient):
        return SimpleNamespace(loss=self.value * coefficient)


class TrainLoopTests(unittest.TestCase):
    def test_accumulation_cycles_short_data_and_clears_gradients(self):
        model = SyntheticLoss()
        optimizer = RecordingOptimizer(model.value)
        data = [{"coefficient": torch.tensor(0.2)}, {"coefficient": torch.tensor(0.4)}]
        with contextlib.redirect_stdout(io.StringIO()):
            train_steps(model, data, optimizer, 2, 3, torch.device("cpu"))
        self.assertEqual(len(optimizer.gradients), 2)
        self.assertAlmostEqual(optimizer.gradients[0], (0.2 + 0.4 + 0.2) / 3, places=6)
        self.assertAlmostEqual(optimizer.gradients[1], (0.4 + 0.2 + 0.4) / 3, places=6)
        self.assertAlmostEqual(model.value.item(), 0.1)  # No weight update occurred.

    def test_nonfinite_loss_never_reaches_optimizer(self):
        model = SyntheticLoss()
        optimizer = RecordingOptimizer(model.value)
        data = [{"coefficient": torch.tensor(float("nan"))}]
        with self.assertRaisesRegex(RuntimeError, "Non-finite loss"):
            train_steps(model, data, optimizer, 1, 1, torch.device("cpu"))
        self.assertEqual(optimizer.gradients, [])


if __name__ == "__main__":
    unittest.main()
