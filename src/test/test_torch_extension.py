from __future__ import annotations

import sys
from pathlib import Path
import unittest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.torch_wrapper import TorchSNNext
from backend.neuro_pyramidal import SynapseParams


class TorchExtSurrogateTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not torch.cuda.is_available():
      raise unittest.SkipTest("CUDA is required.")

  def test_forward_backward_external_current(self):
    model = TorchSNNext(num_neurons=4, dt=1e-4, surrogate_slope=4.0).cuda()
    ext = torch.zeros(4, device="cuda", requires_grad=True)
    spikes, v_soma, v_ais = model(ext)
    self.assertEqual(spikes.shape, (4,))
    loss = spikes.sum()
    loss.backward()
    self.assertIsNotNone(ext.grad)
    self.assertTrue(torch.isfinite(ext.grad).all())

  def test_forward_with_synapse(self):
    model = TorchSNNext(num_neurons=2, dt=1e-4, surrogate_slope=4.0).cuda()
    model.connect([0], [1], SynapseParams(), w_init=1.0)
    ext = torch.zeros(2, device="cuda", requires_grad=True)
    spikes, v_soma, v_ais = model(ext)
    self.assertEqual(spikes.shape, (2,))
    loss = spikes.sum()
    loss.backward()
    self.assertIsNotNone(ext.grad)
    self.assertTrue(torch.isfinite(ext.grad).all())
    self.assertIsNotNone(model.syn_w.grad)
    self.assertTrue(torch.isfinite(model.syn_w.grad).all())

  def test_multi_layer_multi_batch(self):
    layer1 = TorchSNNext(num_neurons=2, dt=1e-4, surrogate_slope=4.0, batch_size=2).cuda()
    layer1.connect([0], [1], SynapseParams(), w_init=1.0)
    layer2 = TorchSNNext(num_neurons=2, dt=1e-4, surrogate_slope=4.0, batch_size=2).cuda()
    layer2.connect([0], [1], SynapseParams(), w_init=1.0)

    ext = torch.zeros((2, 2), device="cuda", requires_grad=True)
    spikes1, _, _ = layer1(ext)
    spikes2, _, _ = layer2(spikes1)
    loss = spikes2.sum()
    loss.backward()

    for syn_w in (layer1.syn_w, layer2.syn_w):
      self.assertIsNotNone(syn_w.grad)
      self.assertTrue(torch.isfinite(syn_w.grad).all())
    self.assertIsNotNone(ext.grad)
    self.assertTrue(torch.isfinite(ext.grad).all())

  def test_multistep_stability(self):
    steps = 100
    layer1 = TorchSNNext(num_neurons=2, dt=1e-4, surrogate_slope=4.0, batch_size=2).cuda()
    layer1.connect([0], [1], SynapseParams(), w_init=1.0)
    layer2 = TorchSNNext(num_neurons=2, dt=1e-4, surrogate_slope=4.0, batch_size=2).cuda()
    layer2.connect([0], [1], SynapseParams(), w_init=1.0)

    ext = torch.zeros((2, 2), device="cuda", requires_grad=True)
    total_loss = 0.0
    for _ in range(steps):
      spikes1, _, _ = layer1(ext)
      spikes2, _, _ = layer2(spikes1)
      total_loss = total_loss + spikes2.sum()
    total_loss.backward()

    for syn_w in (layer1.syn_w, layer2.syn_w):
      self.assertIsNotNone(syn_w.grad)
      self.assertTrue(torch.isfinite(syn_w.grad).all())
    self.assertIsNotNone(ext.grad)
    self.assertTrue(torch.isfinite(ext.grad).all())

if __name__ == "__main__":
  unittest.main()
