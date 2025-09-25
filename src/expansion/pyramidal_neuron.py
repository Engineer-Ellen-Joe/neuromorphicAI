"""
GPU-accelerated pyramidal neuron with synaptic plasticity.

This module provides a detailed implementation of a cortical pyramidal neuron
with explicit modeling of axon initial segment thresholding, axonal branching
with safety factors, and synaptic plasticity mechanisms (STDP and BCM-style
LTP). All stateful arrays reside on the GPU via CuPy and use float64 precision.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cupy as cp
import numpy as np

__all__ = ["PyramidalNeuron", "StepResult"]

DTYPE = cp.float64 # TODO: float64를 float32로 변경

_TRACE_INTEGRATOR = cp.ElementwiseKernel(
    "float64 trace, float64 signal, float64 decay",
    "float64 updated",
    "updated = trace * decay + signal;",
    "trace_integrator_f64",
)
"""
    현재 스크립트의 디렉토리를 가져와 .cu 파일의 전체 경로를 작성합니다.
    스크립트가 실행되는 위치에 관계없이 파일을 찾을 수 있습니다.
"""
script_dir = os.path.dirname(os.path.abspath(__file__))
cu_file_path = os.path.join(script_dir, 'pyramidal_neuron.cu')
with open(cu_file_path, 'r') as f:
    _cuda_source = f.read()

# load kernel
_STDP_KERNEL = cp.RawKernel(_cuda_source, 'stdp_update')
_PROPAGATE_KERNEL = cp.RawKernel(_cuda_source, 'propagate_branch')
_LTP_KERNEL = cp.RawKernel(_cuda_source, 'bcm_ltp')

_DENDRITE_KERNEL = cp.ElementwiseKernel(
    "float64 potential, float64 input_current, float64 dt, float64 tau_inv, float64 capacitance_inv",
    "float64 updated",
    "updated = potential + dt * (-potential * tau_inv + input_current * capacitance_inv);",
    "dendritic_update_f64",
)

@dataclass(frozen=True)
class StepResult:
    """Container for a single simulation step."""

    soma_potential: np.float64
    ais_activation: np.float64
    axon_spike: np.float64
    branch_currents: cp.ndarray
    postsynaptic_drive: cp.ndarray
    postsynaptic_potentials: cp.ndarray

class PyramidalNeuron:
    """Detailed pyramidal neuron with STDP and BCM-style LTP on the GPU."""

    def __init__(
        self,
        num_afferents: int,
        num_branches: int,
        *,
        # TODO: 아래 수치를 오버플로우가 발생하지 않는 float32 맞춤형 수치(비율은 유지)로 변경
        dt: float = 1e-3,
        membrane_time_constant: float = 20e-3,
        membrane_capacitance: float = 200e-12,
        leak_potential: float = -0.07,
        reset_potential: float = -0.065,
        ais_threshold: float = -0.05,
        ais_bias: float = 0.0,
        ais_slope: float = 0.001,
        ais_activation_gate: float = 0.5,
        refractory_period: float = 2e-3,
        axon_spike_current: float = 1.5e-9,
        conduction_threshold: float = 5e-10,
        input_learning_rate: float = 1e-3,
        output_learning_rate: float = 1e-3,
        stdp_a_plus: float = 0.01,
        stdp_a_minus: float = 0.012,
        stdp_tau_pre: float = 20e-3,
        stdp_tau_post: float = 20e-3,
        branch_activity_tau: float = 50e-3,
        bcm_tau: float = 1.0,
        dendritic_time_constant: float = 40e-3,
        dendritic_capacitance: float = 150e-12,
        initial_input_weights: Optional[Sequence[float]] = None,
        initial_output_weights: Optional[Sequence[float]] = None,
        safety_factors: Optional[Sequence[float]] = None,
        input_weight_bounds: Optional[Sequence[float]] = None,
        output_weight_bounds: Optional[Sequence[float]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if num_afferents <= 0:
            raise ValueError("num_afferents must be a positive integer")
        if num_branches <= 0:
            raise ValueError("num_branches must be a positive integer")

        self.num_afferents = int(num_afferents)
        self.num_branches = int(num_branches)

        if random_state is not None:
            cp.random.seed(random_state)

        self.leak_potential = float(leak_potential)
        self.reset_potential = float(reset_potential)
        self.ais_threshold = float(ais_threshold)
        self.ais_bias = float(ais_bias)
        self.ais_slope = float(ais_slope)
        self.ais_activation_gate = float(ais_activation_gate)

        self.membrane_time_constant = float(membrane_time_constant)
        self.membrane_capacitance = float(membrane_capacitance)
        self.refractory_period = float(refractory_period)
        self.refractory_timer = float(0.0)

        self.axon_spike_current = float(axon_spike_current)
        self.conduction_threshold = float(conduction_threshold)

        self.input_learning_rate = float(input_learning_rate)
        self.output_learning_rate = float(output_learning_rate)
        self.stdp_a_plus = float(stdp_a_plus)
        self.stdp_a_minus = float(stdp_a_minus)
        self.stdp_tau_pre = float(stdp_tau_pre)
        self.stdp_tau_post = float(stdp_tau_post)
        self.branch_activity_tau = float(branch_activity_tau)
        self.bcm_tau = float(bcm_tau)
        self.dendritic_time_constant = float(dendritic_time_constant)
        self.dendritic_capacitance = float(dendritic_capacitance)

        if self.stdp_tau_pre <= 0.0 or self.stdp_tau_post <= 0.0:
            raise ValueError("STDP time constants must be positive")
        if self.branch_activity_tau <= 0.0 or self.bcm_tau <= 0.0:
            raise ValueError("LTP time constants must be positive")
        if self.dendritic_time_constant <= 0.0:
            raise ValueError("dendritic_time_constant must be positive")
        if self.dendritic_capacitance <= 0.0:
            raise ValueError("dendritic_capacitance must be positive")

        self._leak_potential_gpu = DTYPE(self.leak_potential)
        self._reset_potential_gpu = DTYPE(self.reset_potential)
        self._ais_threshold_gpu = DTYPE(self.ais_threshold)
        self._ais_bias_gpu = DTYPE(self.ais_bias)
        self._ais_slope_gpu = DTYPE(self.ais_slope)
        self._ais_activation_gate_gpu = DTYPE(self.ais_activation_gate)
        self._membrane_time_constant_gpu = DTYPE(self.membrane_time_constant)
        self._membrane_capacitance_gpu = DTYPE(self.membrane_capacitance)
        self._axon_spike_current_gpu = DTYPE(self.axon_spike_current)
        self._conduction_threshold_gpu = DTYPE(self.conduction_threshold)
        self.input_learning_rate_gpu = DTYPE(self.input_learning_rate)
        self.output_learning_rate_gpu = DTYPE(self.output_learning_rate)
        self.stdp_a_plus_gpu = DTYPE(self.stdp_a_plus)
        self.stdp_a_minus_gpu = DTYPE(self.stdp_a_minus)
        self.bcm_tau_inv_gpu = DTYPE(1.0 / self.bcm_tau)
        self._dendritic_tau_inv_gpu = DTYPE(1.0 / self.dendritic_time_constant)
        self._dendritic_capacitance_gpu = DTYPE(self.dendritic_capacitance)
        self._dendritic_capacitance_inv_gpu = DTYPE(1.0 / self.dendritic_capacitance)

        if input_weight_bounds is None:
            input_weight_bounds = (0.0, 1.0)
        if output_weight_bounds is None:
            output_weight_bounds = (0.0, 1.0)
        if len(input_weight_bounds) != 2:
            raise ValueError("input_weight_bounds must contain (min, max)")
        if len(output_weight_bounds) != 2:
            raise ValueError("output_weight_bounds must contain (min, max)")
        self.input_weight_bounds = (
            DTYPE(float(input_weight_bounds[0])),
            DTYPE(float(input_weight_bounds[1])),
        )
        self.output_weight_bounds = (
            DTYPE(float(output_weight_bounds[0])),
            DTYPE(float(output_weight_bounds[1])),
        )

        if initial_input_weights is None:
            self.input_weights = cp.random.uniform(
                float(self.input_weight_bounds[0]),
                float(self.input_weight_bounds[1]),
                size=self.num_afferents,
            ).astype(DTYPE)
        else:
            self.input_weights = self._validate_vector(
                initial_input_weights,
                self.num_afferents,
                "initial_input_weights",
            )

        if initial_output_weights is None:
            self.output_weights = cp.random.uniform(
                float(self.output_weight_bounds[0]),
                float(self.output_weight_bounds[1]),
                size=self.num_branches,
            ).astype(DTYPE)
        else:
            self.output_weights = self._validate_vector(
                initial_output_weights,
                self.num_branches,
                "initial_output_weights",
            )

        if safety_factors is None:
            self.safety_factors = cp.ones(self.num_branches, dtype=DTYPE)
        else:
            self.safety_factors = self._validate_vector(
                safety_factors,
                self.num_branches,
                "safety_factors",
            )
        if cp.any(self.safety_factors <= 0.0):
            raise ValueError("Safety factors must be strictly positive")

        self.pre_trace = cp.zeros(self.num_afferents, dtype=DTYPE)
        self.post_trace = cp.zeros(self.num_afferents, dtype=DTYPE)
        self.postsynaptic_activity_trace = cp.zeros(self.num_branches, dtype=DTYPE)
        self.bcm_theta = cp.zeros(self.num_branches, dtype=DTYPE)
        self.branch_currents = cp.zeros(self.num_branches, dtype=DTYPE)
        self.postsynaptic_potentials = cp.zeros(self.num_branches, dtype=DTYPE)

        self.membrane_potential = cp.full((1,), self.leak_potential, dtype=DTYPE)
        self._update_time_step(dt)

    @staticmethod
    def _validate_vector(data: Sequence[float], length: int, name: str) -> cp.ndarray:
        array = cp.asarray(data, dtype=DTYPE)
        if array.ndim != 1 or array.size != length:
            raise ValueError(f"{name} must be a vector of length {length}")
        return array

    def _update_time_step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self._dt_gpu = DTYPE(self.dt)
        self.pre_decay = DTYPE(math.exp(-self.dt / self.stdp_tau_pre))
        self.post_decay = DTYPE(math.exp(-self.dt / self.stdp_tau_post))
        self.postsynaptic_decay = DTYPE(math.exp(-self.dt / self.branch_activity_tau))

    def reset_state(self) -> None:
        self.membrane_potential.fill(self.leak_potential)
        self.refractory_timer = 0.0
        self.pre_trace.fill(0.0)
        self.post_trace.fill(0.0)
        self.postsynaptic_activity_trace.fill(0.0)
        self.postsynaptic_potentials.fill(0.0)
        self.bcm_theta.fill(0.0)
        self.branch_currents.fill(0.0)

    def step(
        self,
        presynaptic_spikes: Sequence[float],
        *,
        external_current: float = 0.0,
        dt: Optional[float] = None,
    ) -> StepResult:
        if dt is not None:
            self._update_time_step(dt)
        else:
            dt = self.dt

        pre_spikes = cp.asarray(presynaptic_spikes, dtype=DTYPE)
        if pre_spikes.ndim != 1 or pre_spikes.size != self.num_afferents:
            raise ValueError(
                f"presynaptic_spikes must have shape ({self.num_afferents},)"
            )

        self.refractory_timer = max(0.0, self.refractory_timer - float(dt))

        self.pre_trace = _TRACE_INTEGRATOR(self.pre_trace, pre_spikes, self.pre_decay)

        synaptic_drive = cp.dot(self.input_weights, pre_spikes)
        external_current_gpu = DTYPE(float(external_current))

        if self.refractory_timer > 0.0:
            self.membrane_potential.fill(self.reset_potential)
            ais_activation = cp.zeros_like(self.membrane_potential)
            axon_spike_gpu = DTYPE(0.0)
        else:
            leak_term = (
                self._leak_potential_gpu - self.membrane_potential
            ) / self._membrane_time_constant_gpu
            depolarization = self._dt_gpu * (
                leak_term
                + (synaptic_drive + external_current_gpu)
                / self._membrane_capacitance_gpu
            )
            self.membrane_potential = self.membrane_potential + depolarization
            ais_voltage = self.membrane_potential + self._ais_bias_gpu
            ais_activation = 1.0 / (
                1.0
                + cp.exp(
                    -(
                        (ais_voltage - self._ais_threshold_gpu)
                        / self._ais_slope_gpu
                    )
                )
            )
            spike_mask = ais_activation >= self._ais_activation_gate_gpu
            if bool(cp.asnumpy(spike_mask)):
                axon_spike_gpu = DTYPE(1.0)
                self.membrane_potential.fill(self.reset_potential)
                self.refractory_timer = self.refractory_period
            else:
                axon_spike_gpu = DTYPE(0.0)

        post_signal = cp.full(
            self.num_afferents,
            axon_spike_gpu,
            dtype=DTYPE,
        )
        self.post_trace = _TRACE_INTEGRATOR(
            self.post_trace,
            post_signal,
            self.post_decay,
        )

        block_size = 128
        grid_aff = (self.num_afferents + block_size - 1) // block_size
        _STDP_KERNEL(
            (grid_aff,),
            (block_size,),
            (
                self.input_weights,
                self.pre_trace,
                self.post_trace,
                pre_spikes,
                axon_spike_gpu,
                self.stdp_a_plus_gpu,
                self.stdp_a_minus_gpu,
                self.input_weight_bounds[0],
                self.input_weight_bounds[1],
                self.input_learning_rate_gpu,
                np.int32(self.num_afferents),
            ),
        )

        self.branch_currents.fill(0.0)
        grid_br = (self.num_branches + block_size - 1) // block_size
        _PROPAGATE_KERNEL(
            (grid_br,),
            (block_size,),
            (
                axon_spike_gpu,
                self._axon_spike_current_gpu,
                self._conduction_threshold_gpu,
                self.safety_factors,
                self.branch_currents,
                np.int32(self.num_branches),
            ),
        )

        postsynaptic_currents = self.branch_currents * self.output_weights

        self.postsynaptic_activity_trace = _TRACE_INTEGRATOR(
            self.postsynaptic_activity_trace,
            postsynaptic_currents,
            self.postsynaptic_decay,
        )

        self.postsynaptic_potentials = _DENDRITE_KERNEL(
            self.postsynaptic_potentials,
            postsynaptic_currents,
            self._dt_gpu,
            self._dendritic_tau_inv_gpu,
            self._dendritic_capacitance_inv_gpu,
        )

        _LTP_KERNEL(
            (grid_br,),
            (block_size,),
            (
                self.bcm_theta,
                self.output_weights,
                self.postsynaptic_activity_trace,
                self.bcm_tau_inv_gpu,
                self.output_learning_rate_gpu,
                self.output_weight_bounds[0],
                self.output_weight_bounds[1],
                self._dt_gpu,
                np.int32(self.num_branches),
            ),
        )

        branch_currents_copy = self.branch_currents.copy()
        postsynaptic_currents_copy = postsynaptic_currents.copy()
        dendritic_potentials_copy = self.postsynaptic_potentials.copy()

        soma_potential_host = np.float64(cp.asnumpy(self.membrane_potential).item())
        ais_activation_host = np.float64(cp.asnumpy(ais_activation).item())
        axon_spike_host = np.float64(cp.asnumpy(axon_spike_gpu))

        return StepResult(
            soma_potential=soma_potential_host,
            ais_activation=ais_activation_host,
            axon_spike=axon_spike_host,
            branch_currents=branch_currents_copy,
            postsynaptic_drive=postsynaptic_currents_copy,
            postsynaptic_potentials=dendritic_potentials_copy,
        )