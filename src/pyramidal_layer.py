"""
GPU-accelerated pyramidal neuron layer with synaptic plasticity.

This module provides a vectorized implementation of a layer of cortical
pyramidal neurons, designed for large-scale simulations. All stateful arrays
reside on the GPU via CuPy and are intended to be evolved to use float32
precision for performance.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cupy as cp
import numpy as np

__all__ = ["PyramidalLayer", "StepResult"]

DTYPE = cp.float32

# Load CUDA source code from .cu file
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file_path = os.path.join(script_dir, 'pyramidal_layer.cu')
    with open(cu_file_path, 'r') as f:
        _cuda_source = f.read()
    # Define RawKernel objects for the layer operations
    _STDP_LAYER_KERNEL = cp.RawKernel(_cuda_source, 'stdp_update_layer')
    _PROPAGATE_LAYER_KERNEL = cp.RawKernel(_cuda_source, 'propagate_branch_layer')
    _LTP_LAYER_KERNEL = cp.RawKernel(_cuda_source, 'bcm_ltp_layer')
except FileNotFoundError:
    _cuda_source = None
    _STDP_LAYER_KERNEL = None
    _PROPAGATE_LAYER_KERNEL = None
    _LTP_LAYER_KERNEL = None
    print("Warning: pyramidal_layer.cu not found. CUDA kernels will not be available.")

# Vectorized version of the trace integrator
_TRACE_INTEGRATOR_LAYER = cp.ElementwiseKernel(
    "float32 trace, float32 signal, float32 decay",
    "float32 updated",
    "updated = trace * decay + signal;",
    "trace_integrator_layer",
)

_DENDRITE_KERNEL_LAYER = cp.ElementwiseKernel(
    "P pot, C I, float32 dt, float32 tau_inv, float32 cap_inv",
    "P updated",
    "updated = pot + dt * (-pot * tau_inv + I * cap_inv);",
    "dendritic_update_layer",
)

@dataclass(frozen=True)
class StepResult:
    """Container for a single simulation step for the entire layer."""
    soma_potentials: cp.ndarray
    ais_activations: cp.ndarray
    axon_spikes: cp.ndarray
    branch_currents: cp.ndarray
    postsynaptic_drive: cp.ndarray
    postsynaptic_potentials: cp.ndarray

class PyramidalLayer:
    """Vectorized layer of pyramidal neurons with STDP and BCM-style LTP."""

    def __init__(
        self,
        num_neurons: int,
        num_afferents: int,
        num_branches: int,
        *,
        # Parameters (ms, mV, nF, nA)
        dt: float = 1.0,
        membrane_time_constant: float = 20.0,
        membrane_capacitance: float = 0.2,
        leak_potential: float = -70.0,
        reset_potential: float = -65.0,
        ais_threshold: float = -50.0,
        ais_bias: float = 0.0,
        ais_slope: float = 1.0,
        ais_activation_gate: float = 0.5,
        refractory_period: float = 2.0,
        axon_spike_current: float = 1.5,
        conduction_threshold: float = 0.5,
        input_learning_rate: float = 1e-3,
        output_learning_rate: float = 1e-3,
        stdp_a_plus: float = 0.01,
        stdp_a_minus: float = 0.012,
        stdp_tau_pre: float = 20.0,
        stdp_tau_post: float = 20.0,
        branch_activity_tau: float = 50.0,
        bcm_tau: float = 1000.0,
        dendritic_time_constant: float = 40.0,
        dendritic_capacitance: float = 0.15,
        initial_input_weights: Optional[Sequence[Sequence[float]]] = None,
        initial_output_weights: Optional[Sequence[Sequence[float]]] = None,
        safety_factors: Optional[Sequence[Sequence[float]]] = None,
        input_weight_bounds: Optional[Sequence[float]] = None,
        output_weight_bounds: Optional[Sequence[float]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if num_neurons <= 0:
            raise ValueError("num_neurons must be a positive integer")
        if num_afferents <= 0:
            raise ValueError("num_afferents must be a positive integer")
        if num_branches <= 0:
            raise ValueError("num_branches must be a positive integer")

        self.num_neurons = int(num_neurons)
        self.num_afferents = int(num_afferents)
        self.num_branches = int(num_branches)

        if random_state is not None:
            cp.random.seed(random_state)

        # --- Scalar parameters (remain the same) ---
        self.leak_potential = float(leak_potential)
        self.reset_potential = float(reset_potential)
        self.ais_threshold = float(ais_threshold)
        self.ais_bias = float(ais_bias)
        self.ais_slope = float(ais_slope)
        self.ais_activation_gate = float(ais_activation_gate)
        self.refractory_period = float(refractory_period)
        self.axon_spike_current = float(axon_spike_current)
        self.conduction_threshold = float(conduction_threshold)
        self.input_learning_rate = float(input_learning_rate)
        self.output_learning_rate = float(output_learning_rate)

        self.membrane_time_constant = float(membrane_time_constant)
        self.membrane_capacitance = float(membrane_capacitance)
        self.stdp_a_plus = float(stdp_a_plus)
        self.stdp_a_minus = float(stdp_a_minus)
        self.stdp_tau_pre = float(stdp_tau_pre)
        self.stdp_tau_post = float(stdp_tau_post)
        self.branch_activity_tau = float(branch_activity_tau)
        self.bcm_tau = float(bcm_tau)
        self.dendritic_time_constant = float(dendritic_time_constant)
        self.dendritic_capacitance = float(dendritic_capacitance)

        # --- Convert parameters to GPU-friendly DTYPE ---
        # This part remains largely the same, as they are broadcasted during operations
        self._leak_potential_gpu = DTYPE(self.leak_potential)
        self._reset_potential_gpu = DTYPE(self.reset_potential)
        # ... (and so on for other scalar parameters)

        # --- Weight bounds ---
        if input_weight_bounds is None: input_weight_bounds = (0.0, 1.0)
        if output_weight_bounds is None: output_weight_bounds = (0.0, 1.0)
        self.input_weight_bounds = (DTYPE(input_weight_bounds[0]), DTYPE(input_weight_bounds[1]))
        self.output_weight_bounds = (DTYPE(output_weight_bounds[0]), DTYPE(output_weight_bounds[1]))

        # --- Vectorized state variables and parameters ---
        # From now on, the first dimension (size num_neurons) represents the layer

        if initial_input_weights is None:
            self.input_weights = cp.random.uniform(
                float(self.input_weight_bounds[0]),
                float(self.input_weight_bounds[1]),
                size=(self.num_neurons, self.num_afferents),
            ).astype(DTYPE)
        else:
            self.input_weights = self._validate_matrix(
                initial_input_weights,
                (self.num_neurons, self.num_afferents),
                "initial_input_weights",
            )

        if initial_output_weights is None:
            self.output_weights = cp.random.uniform(
                float(self.output_weight_bounds[0]),
                float(self.output_weight_bounds[1]),
                size=(self.num_neurons, self.num_branches),
            ).astype(DTYPE)
        else:
            self.output_weights = self._validate_matrix(
                initial_output_weights,
                (self.num_neurons, self.num_branches),
                "initial_output_weights",
            )

        if safety_factors is None:
            self.safety_factors = cp.ones((self.num_neurons, self.num_branches), dtype=DTYPE)
        else:
            self.safety_factors = self._validate_matrix(
                safety_factors,
                (self.num_neurons, self.num_branches),
                "safety_factors",
            )

        # State variables are now vectors or matrices
        self.membrane_potential = cp.full((self.num_neurons,), self.leak_potential, dtype=DTYPE)
        self.refractory_timers = cp.zeros(self.num_neurons, dtype=DTYPE)

        self.pre_trace = cp.zeros((self.num_neurons, self.num_afferents), dtype=DTYPE)
        self.post_trace = cp.zeros((self.num_neurons, self.num_afferents), dtype=DTYPE)

        self.postsynaptic_activity_trace = cp.zeros((self.num_neurons, self.num_branches), dtype=DTYPE)
        self.bcm_theta = cp.zeros((self.num_neurons, self.num_branches), dtype=DTYPE)
        self.branch_currents = cp.zeros((self.num_neurons, self.num_branches), dtype=DTYPE)
        self.postsynaptic_potentials = cp.zeros((self.num_neurons, self.num_branches), dtype=DTYPE)

        self._update_time_step(dt)

    @staticmethod
    def _validate_matrix(data: Sequence[Sequence[float]], shape: tuple[int, int], name: str) -> cp.ndarray:
        array = cp.asarray(data, dtype=DTYPE)
        if array.shape != shape:
            raise ValueError(f"{name} must be a matrix of shape {shape}")
        return array

    def _update_time_step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self._dt_gpu = DTYPE(self.dt)
        # These decay factors are scalar and will be broadcasted.
        self.pre_decay = DTYPE(math.exp(-self.dt / self.stdp_tau_pre))
        self.post_decay = DTYPE(math.exp(-self.dt / self.stdp_tau_post))
        self.postsynaptic_decay = DTYPE(math.exp(-self.dt / self.branch_activity_tau))

    def reset_state(self) -> None:
        """Resets the state of all neurons in the layer."""
        self.membrane_potential.fill(self.leak_potential)
        self.refractory_timers.fill(0.0)
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.postsynaptic_activity_trace.fill(0.0)
        self.postsynaptic_potentials.fill(0.0)
        self.bcm_theta.fill(0.0)
        self.branch_currents.fill(0.0)

    def step(
        self,
        presynaptic_spikes: cp.ndarray,
        *,
        external_currents: cp.ndarray,
        dt: Optional[float] = None,
    ) -> StepResult:
        if dt is not None:
            self._update_time_step(dt)

        # Ensure inputs are on the correct device and have the right shape
        if presynaptic_spikes.shape != (self.num_afferents,):
            raise ValueError(f"presynaptic_spikes must have shape ({self.num_afferents},)")
        if external_currents.shape != (self.num_neurons,):
            raise ValueError(f"external_currents must have shape ({self.num_neurons},)")

        # 1. Update refractory timers
        self.refractory_timers = cp.maximum(0.0, self.refractory_timers - self.dt)

        # 2. Update presynaptic traces (STDP)
        # presynaptic_spikes are broadcasted across all neurons for this calculation
        self.pre_trace = _TRACE_INTEGRATOR_LAYER(self.pre_trace, presynaptic_spikes, self.pre_decay)

        # 3. Calculate total synaptic drive for each neuron
        # This is the core vectorization step: a matrix-vector multiplication
        synaptic_drive = cp.dot(self.input_weights, presynaptic_spikes)

        # 4. Update membrane potential for non-refractory neurons
        active_mask = self.refractory_timers == 0.0
        
        leak_term = (self._leak_potential_gpu - self.membrane_potential) / self.membrane_time_constant
        
        # Note: capacitance is currently scalar, should it be per-neuron? Assuming scalar for now.
        depolarization = self.dt * (
            leak_term + (synaptic_drive + external_currents) / self.membrane_capacitance
        )
        
        # Only update potentials of non-refractory neurons
        self.membrane_potential += depolarization * active_mask

        # 5. Determine which neurons spike
        ais_voltage = self.membrane_potential + self.ais_bias
        ais_activation = 1.0 / (1.0 + cp.exp(-(ais_voltage - self.ais_threshold) / self.ais_slope))
        
        # Spike if not in refractory period and activation crosses gate
        axon_spikes_bool = (ais_activation >= self.ais_activation_gate) * active_mask
        
        # 6. Reset potentials and set refractory timers for neurons that spiked
        self.membrane_potential[axon_spikes_bool] = self.reset_potential
        self.refractory_timers[axon_spikes_bool] = self.refractory_period
        
        # Convert boolean spikes to float for kernel calculations
        axon_spikes = axon_spikes_bool.astype(DTYPE)
        
        # Neurons in refractory period are clamped to reset potential
        self.membrane_potential[~active_mask] = self.reset_potential

        # --- Plasticity and Postsynaptic calculations ---
        
        # 7. Update post_trace (STDP)
        # Broadcast the axon_spikes vector to the shape of post_trace
        post_signal = cp.broadcast_to(axon_spikes[:, None], self.post_trace.shape)
        self.post_trace = _TRACE_INTEGRATOR_LAYER(self.post_trace, post_signal, self.post_decay)

        # 8. Call STDP CUDA Kernel
        if _STDP_LAYER_KERNEL is not None:
            threads_per_block = (16, 16)
            grid_x = (self.num_afferents + threads_per_block[0] - 1) // threads_per_block[0]
            grid_y = (self.num_neurons + threads_per_block[1] - 1) // threads_per_block[1]
            grid = (grid_x, grid_y)
            
            _STDP_LAYER_KERNEL(
                grid,
                threads_per_block,
                (
                    self.input_weights,
                    self.pre_trace,
                    self.post_trace,
                    presynaptic_spikes,
                    axon_spikes,
                    DTYPE(self.stdp_a_plus),
                    DTYPE(self.stdp_a_minus),
                    self.input_weight_bounds[0],
                    self.input_weight_bounds[1],
                    DTYPE(self.input_learning_rate),
                    np.int32(self.num_neurons),
                    np.int32(self.num_afferents),
                ),
            )

        # 9. Call Axon Branch Propagation CUDA Kernel
        if _PROPAGATE_LAYER_KERNEL is not None:
            threads_per_block = (16, 16)
            grid_x = (self.num_branches + threads_per_block[0] - 1) // threads_per_block[0]
            grid_y = (self.num_neurons + threads_per_block[1] - 1) // threads_per_block[1]
            grid = (grid_x, grid_y)

            _PROPAGATE_LAYER_KERNEL(
                grid,
                threads_per_block,
                (
                    axon_spikes,
                    DTYPE(self.axon_spike_current),
                    DTYPE(self.conduction_threshold),
                    self.safety_factors,
                    self.branch_currents,
                    np.int32(self.num_neurons),
                    np.int32(self.num_branches),
                ),
            )

        # 10. Calculate postsynaptic drive and update traces and potentials
        postsynaptic_drive = self.branch_currents * self.output_weights

        self.postsynaptic_activity_trace = _TRACE_INTEGRATOR_LAYER(
            self.postsynaptic_activity_trace,
            postsynaptic_drive,
            self.postsynaptic_decay,
        )

        self.postsynaptic_potentials = _DENDRITE_KERNEL_LAYER(
            self.postsynaptic_potentials,
            postsynaptic_drive,
            self._dt_gpu,
            DTYPE(1.0 / self.dendritic_time_constant),
            DTYPE(1.0 / self.dendritic_capacitance),
        )

        # 11. Call BCM LTP CUDA Kernel
        if _LTP_LAYER_KERNEL is not None:
            threads_per_block = (16, 16)
            grid_x = (self.num_branches + threads_per_block[0] - 1) // threads_per_block[0]
            grid_y = (self.num_neurons + threads_per_block[1] - 1) // threads_per_block[1]
            grid = (grid_x, grid_y)

            _LTP_LAYER_KERNEL(
                grid,
                threads_per_block,
                (
                    self.bcm_theta,
                    self.output_weights,
                    self.postsynaptic_activity_trace,
                    DTYPE(1.0 / self.bcm_tau),
                    DTYPE(self.output_learning_rate),
                    self.output_weight_bounds[0],
                    self.output_weight_bounds[1],
                    self._dt_gpu,
                    np.int32(self.num_neurons),
                    np.int32(self.num_branches),
                ),
            )

        return StepResult(
            soma_potentials=self.membrane_potential.copy(),
            ais_activations=ais_activation.copy(),
            axon_spikes=axon_spikes.copy(),
            branch_currents=self.branch_currents.copy(),
            postsynaptic_drive=postsynaptic_drive.copy(),
            postsynaptic_potentials=self.postsynaptic_potentials.copy(),
        )
