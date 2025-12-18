from __future__ import annotations

import sys
from pathlib import Path
import unittest
import torch
import cupy as cp

# Add repository src to import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.neuro_pyramidal import CuPyNetwork, SomaParams, AISParams, CouplingParams
from backend.torch_wrapper import TorchCuPyNetwork


class TorchWrapperSurrogateTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not torch.cuda.is_available():
      raise unittest.SkipTest("CUDA is required for this test.")

  def setUp(self):
    # Minimal 3-neuron network with no synapses.
    self.net = CuPyNetwork(dt=1e-4)
    self.net.add_neurons(3, SomaParams(), AISParams(), CouplingParams())
    self.wrapper = TorchCuPyNetwork(self.net, surrogate_slope=4.0)

  def test_forward_and_surrogate_grad(self):
    ext = torch.zeros(3, device="cuda", requires_grad=True)
    spikes, v_soma, v_ais = self.wrapper(ext)

    self.assertEqual(spikes.shape, (3,))
    self.assertEqual(v_soma.shape, (3,))
    self.assertEqual(v_ais.shape, (3,))
    self.assertEqual(spikes.device.type, "cuda")

    loss = spikes.sum()
    loss.backward()

    self.assertIsNotNone(ext.grad)
    self.assertEqual(ext.grad.shape, ext.shape)
    self.assertTrue(torch.isfinite(ext.grad).all())


if __name__ == "__main__":
  unittest.main()
