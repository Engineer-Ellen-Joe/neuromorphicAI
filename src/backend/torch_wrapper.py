from __future__ import annotations

"""
Torch-facing wrapper around the CuPy-based SNN core.

This keeps the custom CUDA kernels in `backend.neuro_pyramidal` unchanged while
exposing a minimal `nn.Module` interface. Gradients are not propagated through
the CuPy network; this is intended for inference or for use as a fixed SNN
inside a larger Torch model.
"""

import torch
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

from .neuro_pyramidal import CuPyNetwork, SynapseParams, SomaParams, AISParams, CouplingParams
from .torch_extension import load_ext


def _torch_to_cupy(tensor: torch.Tensor) -> cp.ndarray:
  """Zero-copy Torch CUDA tensor -> CuPy array conversion."""
  if not tensor.is_cuda:
    raise ValueError("Expected CUDA tensor for Torch->CuPy conversion.")
  return cp.from_dlpack(to_dlpack(tensor.contiguous()))


def _cupy_to_torch(array: cp.ndarray) -> torch.Tensor:
  """Zero-copy CuPy array -> Torch CUDA tensor conversion."""
  return from_dlpack(array.toDlpack())


class _CuPyStepFn(torch.autograd.Function):
  """
  Runs one CuPyNetwork step and returns Torch tensors.
  Backward uses a sigmoid surrogate on AIS voltage to route grad to external_current.
  """

  @staticmethod
  def forward(ctx, external_current: torch.Tensor | None, network: CuPyNetwork, surrogate_slope: float):
    ctx.surrogate_slope = float(surrogate_slope)
    ctx.batch_size = network.batch_size
    ctx.neurons_per_batch = network.neurons_per_batch
    original_shape = None

    # Optionally update external drive for this step.
    if external_current is not None:
      if not external_current.is_cuda:
        raise ValueError("external_current must be a CUDA tensor.")
      if external_current.ndim == 1:
        if external_current.numel() != network.neurons_per_batch and external_current.numel() != network.n_neurons:
          raise ValueError(f"external_current length must be {network.neurons_per_batch} or {network.n_neurons}")
        if external_current.numel() == network.neurons_per_batch and network.batch_size > 1:
          external_current = external_current.expand(network.batch_size, -1)
      elif external_current.ndim == 2:
        if external_current.shape != (network.batch_size, network.neurons_per_batch):
          raise ValueError(f"external_current must have shape ({network.batch_size}, {network.neurons_per_batch})")
      else:
        raise ValueError("external_current must be 1D or 2D tensor.")

      original_shape = tuple(external_current.shape)
      flat = external_current.reshape(-1)
      target = network.neuron_params["soma_I_ext"]
      src = _torch_to_cupy(flat.to(dtype=torch.float64))
      if target.shape != src.shape:
        raise ValueError(f"external_current shape {src.shape} does not match existing soma_I_ext {target.shape}")
      target[...] = src
    ctx.external_shape = original_shape

    # Advance one simulation step.
    network.step()

    # Ensure kernels are complete before handing buffers to Torch.
    cp.cuda.get_current_stream().synchronize()

    spikes = _cupy_to_torch(network.comm_arrays["spiked_now"]).to(dtype=torch.float32).clone()
    v_soma = _cupy_to_torch(network.neuron_states["V_soma"]).to(dtype=torch.float32).clone()
    v_ais = _cupy_to_torch(network.neuron_states["V_ais"]).to(dtype=torch.float32).clone()

    if network.batch_size > 1:
      spikes = spikes.view(network.batch_size, network.neurons_per_batch)
      v_soma = v_soma.view(network.batch_size, network.neurons_per_batch)
      v_ais = v_ais.view(network.batch_size, network.neurons_per_batch)

    # Surrogate uses AIS voltage; save soft spike for backward.
    threshold = torch.as_tensor(network.neuron_params["ais_V_T"].get(), device=v_ais.device, dtype=v_ais.dtype)
    if network.batch_size > 1:
      threshold = threshold.view(network.batch_size, network.neurons_per_batch)
    soft = torch.sigmoid((v_ais - threshold) / ctx.surrogate_slope)
    ctx.save_for_backward(soft, spikes)
    return spikes, v_soma, v_ais

  @staticmethod
  def backward(ctx, grad_spikes, grad_v_soma, grad_v_ais):
    soft, spikes = ctx.saved_tensors
    slope = ctx.surrogate_slope
    surrogate_grad = soft * (1 - soft) / slope
    grad_external = None
    if grad_spikes is not None:
      grad_external = grad_spikes * surrogate_grad
      if ctx.external_shape is not None:
        grad_external = grad_external.reshape(ctx.external_shape)
    # external_current grad only; network and slope receive no grad.
    return grad_external, None, None


class TorchCuPyNetwork(torch.nn.Module):
  """
  Wraps a stateful CuPyNetwork for use in PyTorch graphs.

  - forward(external_current): advances the SNN by one step.
  - external_current is optional; if provided, it overwrites soma_I_ext.
  - Returns (spikes, V_soma, V_ais) as Torch CUDA tensors (no grad).
  """

  def __init__(self, network: CuPyNetwork, surrogate_slope: float = 4.0):
    super().__init__()
    self.network = network
    self.surrogate_slope = float(surrogate_slope)

  def forward(self, external_current: torch.Tensor | None = None):
    return _CuPyStepFn.apply(external_current, self.network, self.surrogate_slope)


class _ExtStepFn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, external_current: torch.Tensor | None, syn_w: torch.Tensor, wrapper: "TorchSNNext"):
    ext = load_ext()
    if external_current is not None:
      if not external_current.is_cuda:
        raise ValueError("external_current must be CUDA tensor.")
      if external_current.ndim == 2:
        if external_current.shape != (wrapper.batch_size, wrapper.neurons_per_batch):
          raise ValueError(f"external_current must have shape ({wrapper.batch_size}, {wrapper.neurons_per_batch})")
        flat = external_current.reshape(-1).to(dtype=torch.float64)
      else:
        flat = external_current.reshape(-1).to(dtype=torch.float64)
        if flat.numel() != wrapper.n_neurons:
          raise ValueError(f"external_current length must be {wrapper.n_neurons}")
      wrapper.soma_I_ext.copy_(flat)

    out = ext.forward(
      wrapper.V_soma, wrapper.V_ais, wrapper.refrac_until,
      wrapper.I_syn_total, wrapper.spiked_now,
      wrapper.syn_g, syn_w, wrapper.syn_pre_trace, wrapper.syn_post_trace,
      wrapper.syn_r_pre, wrapper.syn_r_post, wrapper.syn_theta_m,
      wrapper.g_incr_buffer,
      wrapper.soma_C_m, wrapper.soma_g_L, wrapper.soma_E_L, wrapper.soma_I_ext,
      wrapper.ais_C_m, wrapper.ais_g_L, wrapper.ais_E_L, wrapper.ais_V_T,
      wrapper.ais_Delta_T, wrapper.ais_V_spike, wrapper.ais_V_reset, wrapper.ais_t_refrac,
      wrapper.coup_g_c,
      wrapper.syn_g_max, wrapper.syn_E_rev, wrapper.syn_tau_decay,
      wrapper.syn_A_plus, wrapper.syn_A_minus, wrapper.syn_tau_pre, wrapper.syn_tau_post,
      wrapper.syn_eta_bcm, wrapper.syn_tau_rate, wrapper.syn_tau_theta,
      wrapper.syn_w_min, wrapper.syn_w_max, wrapper.syn_delay,
      wrapper.syn_pre_idx, wrapper.syn_post_idx,
      int(wrapper.buffer_idx.item()), wrapper.t, wrapper.dt,
    )
    spikes, v_soma, v_ais, soft, next_idx, pre_trace, post_trace, r_pre, r_post, theta_m = out
    wrapper.buffer_idx.copy_(next_idx)
    wrapper.t += wrapper.dt
    ctx.save_for_backward(soft, spikes.float(), v_soma.float(), pre_trace, post_trace, r_pre, r_post, theta_m)
    ctx.surrogate_slope = wrapper.surrogate_slope
    ctx.external_shape = None if external_current is None else tuple(external_current.shape)
    ctx.syn_pre_idx = wrapper.syn_pre_idx
    ctx.syn_post_idx = wrapper.syn_post_idx
    ctx.syn_g_max = wrapper.syn_g_max
    ctx.syn_E_rev = wrapper.syn_E_rev
    ctx.syn_A_plus = wrapper.syn_A_plus
    ctx.syn_A_minus = wrapper.syn_A_minus
    ctx.syn_eta_bcm = wrapper.syn_eta_bcm
    ctx.dt = wrapper.dt
    return spikes.float(), v_soma.float(), v_ais.float()

  @staticmethod
  def backward(ctx, grad_spikes, grad_v_soma, grad_v_ais):
    ext = load_ext()
    soft, spikes, v_soma, pre_trace, post_trace, r_pre, r_post, theta_m = ctx.saved_tensors
    grad_external, grad_w = ext.backward(
      grad_spikes.contiguous(),
      soft,
      spikes,
      v_soma,
      pre_trace,
      post_trace,
      r_pre,
      r_post,
      theta_m,
      ctx.syn_pre_idx,
      ctx.syn_post_idx,
      ctx.syn_g_max,
      ctx.syn_E_rev,
      ctx.syn_A_plus,
      ctx.syn_A_minus,
      ctx.syn_eta_bcm,
      ctx.dt,
      ctx.surrogate_slope,
    )
    if ctx.external_shape is not None:
      grad_external = grad_external.reshape(ctx.external_shape)
    return grad_external, grad_w, None


class TorchSNNext(torch.nn.Module):
  """
  Torch wrapper using C++/CUDA extension.
  Note: Only external_current receives surrogate gradients; weight grads not implemented yet.
  """
  def __init__(self, num_neurons: int, dt: float = 1e-4, surrogate_slope: float = 4.0, batch_size: int = 1,
               soma_params: SomaParams | None = None, ais_params: AISParams | None = None, coup_params: CouplingParams | None = None):
    super().__init__()
    device = torch.device("cuda")
    n = num_neurons
    self.neurons_per_batch = n
    self.batch_size = int(max(1, batch_size))
    self.n_neurons = n * self.batch_size
    self.dt = float(dt)
    self.t = 0.0
    self.surrogate_slope = float(surrogate_slope)
    soma_p = soma_params or SomaParams()
    ais_p = ais_params or AISParams()
    coup_p = coup_params or CouplingParams()

    def _full(val):
      return torch.full((self.n_neurons,), float(val), device=device, dtype=torch.float64, requires_grad=False)

    # Neuron params (define before states so we can seed resting voltages)
    self.soma_C_m = _full(soma_p.C_m)
    self.soma_g_L = _full(soma_p.g_L)
    self.soma_E_L = _full(soma_p.E_L)
    self.soma_I_ext = torch.zeros(self.n_neurons, device=device, dtype=torch.float64, requires_grad=False)

    self.ais_C_m = _full(ais_p.C_m)
    self.ais_g_L = _full(ais_p.g_L)
    self.ais_E_L = _full(ais_p.E_L)
    self.ais_V_T = _full(ais_p.V_T)
    self.ais_Delta_T = _full(ais_p.Delta_T)
    self.ais_V_spike = _full(ais_p.V_spike)
    self.ais_V_reset = _full(ais_p.V_reset)
    self.ais_t_refrac = _full(ais_p.t_refrac)
    self.coup_g_c = _full(coup_p.g_c)

    # States (seed at rest to avoid explosive exp() term)
    self.V_soma = self.soma_E_L.clone()
    self.V_ais = self.ais_E_L.clone()
    self.refrac_until = torch.full((self.n_neurons,), -1e9, device=device, dtype=torch.float64, requires_grad=False)
    self.I_syn_total = torch.zeros(self.n_neurons, device=device, dtype=torch.float64, requires_grad=False)
    self.spiked_now = torch.zeros(self.n_neurons, device=device, dtype=torch.int32, requires_grad=False)

    # Synapse placeholders (will be replaced by connect)
    self.n_synapses = 0
    self.synapses_per_batch = 0
    self._init_empty_synapses(device)

  def _init_empty_synapses(self, device: torch.device):
    self.syn_g = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_w = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False))
    self.syn_pre_trace = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_post_trace = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_r_pre = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_r_post = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_theta_m = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_pre_idx = torch.zeros(1, device=device, dtype=torch.int32, requires_grad=False)
    self.syn_post_idx = torch.zeros(1, device=device, dtype=torch.int32, requires_grad=False)
    self.syn_g_max = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_E_rev = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_tau_decay = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_A_plus = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_A_minus = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_tau_pre = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_tau_post = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_eta_bcm = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_tau_rate = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_tau_theta = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_w_min = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_w_max = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False)
    self.syn_delay = torch.ones(1, device=device, dtype=torch.float64, requires_grad=False) * self.dt
    self.g_incr_buffer = torch.zeros((1, self.syn_g.numel()), device=device, dtype=torch.float64, requires_grad=False)
    self.buffer_idx = torch.zeros(1, device=device, dtype=torch.int64, requires_grad=False)

  def connect(self, pre_indices: list, post_indices: list, syn_params: SynapseParams, w_init: float | torch.Tensor | list = 1.0):
    if len(pre_indices) != len(post_indices):
      raise ValueError("pre_indices and post_indices must have the same length")
    num_synapses_one = len(pre_indices)
    if num_synapses_one == 0:
      self.n_synapses = 0
      self.synapses_per_batch = 0
      self._init_empty_synapses(self.V_soma.device)
      return

    self.synapses_per_batch = num_synapses_one
    self.n_synapses = num_synapses_one * self.batch_size

    # Connectivity replication per batch
    conn_pre = []
    conn_post = []
    for b in range(self.batch_size):
      offset = b * self.neurons_per_batch
      conn_pre.append(torch.tensor(pre_indices, device=self.V_soma.device, dtype=torch.int32) + offset)
      conn_post.append(torch.tensor(post_indices, device=self.V_soma.device, dtype=torch.int32) + offset)
    self.syn_pre_idx = torch.cat(conn_pre)
    self.syn_post_idx = torch.cat(conn_post)

    # Params
    def _tile_param(val):
      base = torch.full((num_synapses_one,), float(val), device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
      return base.repeat(self.batch_size)

    self.syn_g_max = _tile_param(syn_params.g_max)
    self.syn_E_rev = _tile_param(syn_params.E_rev)
    self.syn_tau_decay = _tile_param(syn_params.tau_decay)
    self.syn_A_plus = _tile_param(syn_params.A_plus)
    self.syn_A_minus = _tile_param(syn_params.A_minus)
    self.syn_tau_pre = _tile_param(syn_params.tau_pre)
    self.syn_tau_post = _tile_param(syn_params.tau_post)
    self.syn_eta_bcm = _tile_param(syn_params.eta_bcm)
    self.syn_tau_rate = _tile_param(syn_params.tau_rate)
    self.syn_tau_theta = _tile_param(syn_params.tau_theta)
    self.syn_w_min = _tile_param(syn_params.w_min)
    self.syn_w_max = _tile_param(syn_params.w_max)
    self.syn_delay = _tile_param(syn_params.delay)

    # States
    if isinstance(w_init, (float, int)):
      w_init_arr = torch.full((num_synapses_one,), float(w_init), device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    else:
      w_init_arr = torch.tensor(w_init, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
      if w_init_arr.numel() != num_synapses_one:
        raise ValueError("w_init length must match number of synapses per batch")
    w_init_arr = w_init_arr.repeat(self.batch_size)

    self.syn_g = torch.zeros(self.n_synapses, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    self.syn_w = torch.nn.Parameter(w_init_arr, requires_grad=True)
    self.syn_pre_trace = torch.zeros(self.n_synapses, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    self.syn_post_trace = torch.zeros(self.n_synapses, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    self.syn_r_pre = torch.zeros(self.n_synapses, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    self.syn_r_post = torch.zeros(self.n_synapses, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    self.syn_theta_m = torch.full((self.n_synapses,), 1.0, device=self.V_soma.device, dtype=torch.float64, requires_grad=False)

    # Delay buffer depth by max delay
    delay_steps = max(1, int(torch.ceil(self.syn_delay.max() / self.dt).item()))
    self.g_incr_buffer = torch.zeros((delay_steps, self.n_synapses), device=self.V_soma.device, dtype=torch.float64, requires_grad=False)
    self.buffer_idx = torch.zeros(1, device=self.V_soma.device, dtype=torch.int64, requires_grad=False)

  def forward(self, external_current: torch.Tensor | None = None):
    return _ExtStepFn.apply(external_current, self.syn_w, self)
