"""Unit tests for the pyramidal neuron module."""

import math
import unittest

import sys, os
import cupy as cp
import numpy as np


# 프로젝트 루트를 경로에 추가하여 src 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_neuron import PyramidalNeuron

try:
    _CUDA_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = _CUDA_DEVICE_COUNT > 0
except cp.cuda.runtime.CUDARuntimeError:
    GPU_AVAILABLE = False


def _to_numpy(array):
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def _scalar(array) -> float:
    return float(_to_numpy(array).reshape(-1)[0])


@unittest.skipUnless(GPU_AVAILABLE, "No CUDA-capable device detected")
class PyramidalNeuronTests(unittest.TestCase):
    def test_passive_integration_without_spike(self) -> None:
        neuron = PyramidalNeuron(
            num_afferents=1,
            num_branches=1,
            dt=1e-3,
            initial_input_weights=[5e-9],
            initial_output_weights=[0.5],
            safety_factors=[1.0],
            axon_spike_current=1.0,
            conduction_threshold=2.0,
            output_learning_rate=0.0,
            input_learning_rate=0.0,
        )

        result = neuron.step([0.0])

        self.assertAlmostEqual(result.soma_potential, neuron.leak_potential, places=15)
        self.assertLess(result.ais_activation, neuron.ais_activation_gate)
        self.assertEqual(result.axon_spike, np.float64(0.0))
        np.testing.assert_allclose(_to_numpy(result.branch_currents), 0.0)
        np.testing.assert_allclose(_to_numpy(result.postsynaptic_drive), 0.0)

    def test_single_spike_triggers_branch_conduction(self) -> None:
        neuron = PyramidalNeuron(
            num_afferents=1,
            num_branches=1,
            dt=1e-3,
            initial_input_weights=[5e-9],
            initial_output_weights=[0.8],
            safety_factors=[0.9],
            axon_spike_current=2.0,
            conduction_threshold=0.1,
            output_learning_rate=0.0,
            input_learning_rate=0.0,
        )

        result = neuron.step([1.0])

        self.assertEqual(result.axon_spike, np.float64(1.0))
        expected_branch = neuron.axon_spike_current * _scalar(neuron.safety_factors)
        np.testing.assert_allclose(_to_numpy(result.branch_currents), expected_branch)
        weight_after = _scalar(neuron.output_weights)
        expected_drive = expected_branch * weight_after
        np.testing.assert_allclose(_to_numpy(result.postsynaptic_drive), expected_drive)

    def test_stdp_weight_updates_follow_expected_dynamics(self) -> None:
        neuron = PyramidalNeuron(
            num_afferents=1,
            num_branches=1,
            dt=1e-3,
            initial_input_weights=[0.5],
            initial_output_weights=[0.5],
            safety_factors=[1.0],
            axon_spike_current=1.0,
            conduction_threshold=0.1,
            stdp_a_plus=0.02,
            stdp_a_minus=0.01,
            input_learning_rate=0.1,
            output_learning_rate=0.0,
        )

        neuron.step([1.0])
        weight_after_first_step = _scalar(neuron.input_weights)

        expected_first = 0.5 + neuron.input_learning_rate * (
            neuron.stdp_a_plus * 1.0 * 1.0 - neuron.stdp_a_minus * 1.0 * 1.0
        )
        self.assertAlmostEqual(weight_after_first_step, expected_first, places=10)

        neuron.step([1.0])
        weight_after_second_step = _scalar(neuron.input_weights)

        post_decay = math.exp(-neuron.dt / neuron.stdp_tau_post)
        expected_second = expected_first - neuron.input_learning_rate * neuron.stdp_a_minus * post_decay
        self.assertAlmostEqual(weight_after_second_step, expected_second, places=10)

    def test_bcm_ltp_updates_output_weight(self) -> None:
        neuron = PyramidalNeuron(
            num_afferents=1,
            num_branches=1,
            dt=1e-3,
            initial_input_weights=[5e-9],
            initial_output_weights=[0.5],
            safety_factors=[1.0],
            axon_spike_current=10.0,
            conduction_threshold=0.5,
            branch_activity_tau=5e-3,
            bcm_tau=0.1,
            output_learning_rate=0.5,
            stdp_a_plus=0.0,
            stdp_a_minus=0.0,
            input_learning_rate=0.0,
        )

        result = neuron.step([1.0])

        activity = _scalar(result.branch_currents)
        theta_expected = neuron.dt * (1.0 / neuron.bcm_tau) * activity * activity
        self.assertAlmostEqual(_scalar(neuron.bcm_theta), theta_expected, places=10)

        weight_expected = 0.5 + neuron.output_learning_rate * activity * (
            activity - theta_expected
        ) * neuron.dt
        self.assertAlmostEqual(_scalar(neuron.output_weights), weight_expected, places=10)
        expected_drive = activity * weight_expected
        np.testing.assert_allclose(_to_numpy(result.postsynaptic_drive), expected_drive)

    def test_state_reset_clears_dynamics(self) -> None:
        neuron = PyramidalNeuron(
            num_afferents=1,
            num_branches=1,
            dt=1e-3,
            initial_input_weights=[5e-9],
            initial_output_weights=[0.5],
            safety_factors=[1.0],
            axon_spike_current=2.0,
            conduction_threshold=0.1,
        )

        neuron.step([1.0])
        neuron.reset_state()

        self.assertAlmostEqual(_scalar(neuron.membrane_potential), neuron.leak_potential, places=15)
        np.testing.assert_allclose(_to_numpy(neuron.pre_trace), 0.0)
        np.testing.assert_allclose(_to_numpy(neuron.post_trace), 0.0)
        np.testing.assert_allclose(_to_numpy(neuron.branch_activity_trace), 0.0)
        np.testing.assert_allclose(_to_numpy(neuron.bcm_theta), 0.0)
        np.testing.assert_allclose(_to_numpy(neuron.branch_currents), 0.0)


if __name__ == "__main__":
    unittest.main()
