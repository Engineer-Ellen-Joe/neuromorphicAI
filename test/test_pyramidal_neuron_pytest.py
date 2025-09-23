import math
from typing import List

import sys, os
import cupy as cp
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_neuron import PyramidalNeuron

try:
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - CuPy raises when CUDA unavailable
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason='CuPy GPU device required for pyramidal neuron tests',
)

BASE_PARAMS = dict(
    num_afferents=2,
    num_branches=2,
    dt=1e-3,
    membrane_time_constant=15e-3,
    membrane_capacitance=180e-12,
    leak_potential=-0.07,
    reset_potential=-0.068,
    ais_threshold=-0.062,
    ais_bias=0.0,
    ais_slope=7e-4,
    ais_activation_gate=0.3,
    refractory_period=1e-3,
    axon_spike_current=1.2e-9,
    conduction_threshold=3e-10,
    input_learning_rate=5e-3,
    output_learning_rate=5e-3,
    stdp_a_plus=0.01,
    stdp_a_minus=0.012,
    stdp_tau_pre=15e-3,
    stdp_tau_post=15e-3,
    branch_activity_tau=40e-3,
    bcm_tau=0.6,
    dendritic_time_constant=35e-3,
    dendritic_capacitance=160e-12,
    initial_input_weights=[0.5, 0.4],
    initial_output_weights=[0.7, 0.7],
    safety_factors=[1.1, 0.6],
    input_weight_bounds=(0.0, 1.5),
    output_weight_bounds=(0.0, 1.5),
    random_state=7,
)

def make_neuron(**overrides: float) -> PyramidalNeuron:
    params = BASE_PARAMS.copy()
    params.update(overrides)
    return PyramidalNeuron(**params)

def _to_numpy(array: cp.ndarray) -> np.ndarray:
    return cp.asnumpy(array)

def test_constructor_validations() -> None:
    with pytest.raises(ValueError):
        PyramidalNeuron(num_afferents=0, num_branches=1)
    with pytest.raises(ValueError):
        PyramidalNeuron(num_afferents=1, num_branches=0)
    with pytest.raises(ValueError):
        PyramidalNeuron(num_afferents=1, num_branches=1, stdp_tau_pre=0.0)
    with pytest.raises(ValueError):
        PyramidalNeuron(num_afferents=1, num_branches=1, safety_factors=[0.0])
    with pytest.raises(ValueError):
        PyramidalNeuron(
            num_afferents=2,
            num_branches=1,
            initial_input_weights=[0.1],
        )
    with pytest.raises(ValueError):
        PyramidalNeuron._validate_vector([0.1], 2, 'mismatch')

def test_reset_state_clears_buffers() -> None:
    neuron = make_neuron()
    neuron.step([1.0, 0.0], external_current=1.0e-9)
    neuron.step([0.0, 1.0], external_current=0.0)

    neuron.reset_state()

    assert cp.allclose(neuron.membrane_potential, neuron.leak_potential)
    assert neuron.refractory_timer == pytest.approx(0.0)
    assert cp.allclose(neuron.pre_trace, 0.0)
    assert cp.allclose(neuron.post_trace, 0.0)
    assert cp.allclose(neuron.postsynaptic_activity_trace, 0.0)
    assert cp.allclose(neuron.postsynaptic_potentials, 0.0)
    assert cp.allclose(neuron.bcm_theta, 0.0)
    assert cp.allclose(neuron.branch_currents, 0.0)

def test_dt_override_updates_internal_state() -> None:
    neuron = make_neuron()
    initial_decay = float(neuron.pre_decay)
    neuron.step([0.0, 0.0], external_current=0.0, dt=5e-4)

    assert neuron.dt == pytest.approx(5e-4)
    expected_decay = math.exp(-neuron.dt / neuron.stdp_tau_pre)
    assert float(neuron.pre_decay) == pytest.approx(expected_decay)
    assert float(neuron.pre_decay) != pytest.approx(initial_decay)

def test_spike_generation_and_refractory_behavior() -> None:
    neuron = make_neuron(
        num_afferents=1,
        num_branches=1,
        initial_input_weights=[0.5],
        initial_output_weights=[0.7],
        safety_factors=[1.0],
        conduction_threshold=2e-10,
    )
    neuron.reset_state()

    first = neuron.step([1.0], external_current=1.2e-9)
    assert float(first.axon_spike) == pytest.approx(1.0)
    assert neuron.refractory_timer == pytest.approx(neuron.refractory_period)

    second = neuron.step([0.0], external_current=0.0)
    assert float(second.axon_spike) == pytest.approx(0.0)
    vm = float(_to_numpy(neuron.membrane_potential)[0])
    assert neuron.leak_potential <= vm <= neuron.reset_potential

def test_branch_conduction_respects_safety_factors() -> None:
    neuron = make_neuron(
        num_branches=2,
        initial_output_weights=[0.7, 0.7],
        safety_factors=[1.0, 0.2],
        conduction_threshold=4e-10,
        axon_spike_current=1.2e-9,
    )
    neuron.reset_state()

    result = neuron.step([1.0, 1.0], external_current=1.2e-9)

    branch_currents = _to_numpy(result.branch_currents)
    assert branch_currents[0] == pytest.approx(neuron.axon_spike_current / neuron.num_branches, rel=1e-6)
    assert branch_currents[1] == pytest.approx(0.0, abs=1e-12)

    assert _to_numpy(result.postsynaptic_potentials)[0] > 0.0
    assert neuron.postsynaptic_activity_trace[0] > 0.0

def test_stdp_weight_updates_and_bounds() -> None:
    neuron = make_neuron(
        num_afferents=1,
        num_branches=1,
        initial_input_weights=[0.5],
        initial_output_weights=[0.7],
        safety_factors=[1.0],
        input_weight_bounds=(0.2, 0.8),
        input_learning_rate=2e-2,
        conduction_threshold=2e-10,
    )
    neuron.reset_state()

    neuron.input_weights = cp.full_like(neuron.input_weights, 0.5)
    neuron.step([0.0], external_current=1.2e-9)
    neuron.step([1.0], external_current=0.0)
    ltd_weight = float(_to_numpy(neuron.input_weights)[0])
    assert 0.2 <= ltd_weight < 0.5

    neuron.reset_state()
    neuron.input_weights = cp.full_like(neuron.input_weights, 0.5)
    neuron.step([1.0], external_current=0.0)
    neuron.step([0.0], external_current=1.2e-9)
    ltp_weight = float(_to_numpy(neuron.input_weights)[0])
    assert 0.5 < ltp_weight <= 0.8

    neuron.input_weights = cp.full_like(neuron.input_weights, 0.8)
    neuron.step([1.0], external_current=1.2e-9)
    assert float(_to_numpy(neuron.input_weights)[0]) <= 0.8

    neuron.input_weights = cp.full_like(neuron.input_weights, 0.2)
    neuron.step([0.0], external_current=0.0)
    assert float(_to_numpy(neuron.input_weights)[0]) >= 0.2

def test_bcm_plasticity_and_bounds() -> None:
    neuron = make_neuron(
        num_afferents=1,
        num_branches=2,
        initial_input_weights=[0.5],
        initial_output_weights=[0.6, 0.6],
        safety_factors=[1.0, 0.9],
        output_weight_bounds=(0.3, 1.0),
        output_learning_rate=2e-2,
        axon_spike_current=1.0,
        conduction_threshold=1e-4,
    )
    neuron.reset_state()
    initial_weights = _to_numpy(neuron.output_weights).copy()

    for _ in range(120):
        result = neuron.step([1.0], external_current=1.2e-9)
        assert result.axon_spike == pytest.approx(1.0)

    updated = _to_numpy(neuron.output_weights)
    assert np.any(np.abs(updated - initial_weights) > 1e-4)
    assert np.all(updated <= 1.0 + 1e-12)
    assert np.all(updated >= 0.3 - 1e-12)
    assert _to_numpy(neuron.bcm_theta).max() > 0.0

def test_random_seed_reproducibility() -> None:
    first = make_neuron(initial_input_weights=None, initial_output_weights=None, random_state=42)
    second = make_neuron(initial_input_weights=None, initial_output_weights=None, random_state=42)
    third = make_neuron(initial_input_weights=None, initial_output_weights=None, random_state=43)

    assert cp.allclose(first.input_weights, second.input_weights)
    assert cp.allclose(first.output_weights, second.output_weights)
    assert not cp.allclose(first.input_weights, third.input_weights)

def test_large_cluster_signal_propagation() -> None:
    num_neurons = 100
    branches = 3
    afferents = branches + 1

    neurons: List[PyramidalNeuron] = []
    for idx in range(num_neurons):
        neurons.append(
            PyramidalNeuron(
                num_afferents=afferents,
                num_branches=branches,
                dt=1e-3,
                membrane_time_constant=12e-3,
                membrane_capacitance=180e-12,
                leak_potential=-0.07,
                reset_potential=-0.068,
                ais_threshold=-0.061,
                ais_bias=0.0,
                ais_slope=8e-4,
                ais_activation_gate=0.3,
                refractory_period=1e-3,
                axon_spike_current=1.1e-9,
                conduction_threshold=2.5e-10,
                input_learning_rate=2e-3,
                output_learning_rate=2e-3,
                stdp_a_plus=0.008,
                stdp_a_minus=0.009,
                stdp_tau_pre=12e-3,
                stdp_tau_post=12e-3,
                branch_activity_tau=40e-3,
                bcm_tau=0.6,
                dendritic_time_constant=35e-3,
                dendritic_capacitance=160e-12,
                initial_input_weights=[0.4] * afferents,
                initial_output_weights=[0.6] * branches,
                safety_factors=[1.0, 0.8, 0.6],
                input_weight_bounds=(0.0, 1.0),
                output_weight_bounds=(0.0, 1.2),
                random_state=idx,
            )
        )

    incoming_map: List[List[tuple[int, int]]] = [[] for _ in range(num_neurons)]
    for src in range(num_neurons):
        for branch in range(branches):
            target = (src + branch + 1) % num_neurons
            incoming_map[target].append((src, branch))

    for mapping in incoming_map:
        assert len(mapping) == branches

    previous_branch_currents = [np.zeros(branches, dtype=np.float64) for _ in range(num_neurons)]
    total_spikes = 0.0
    propagation_hits = np.zeros(num_neurons, dtype=np.float64)

    steps = 12
    for step_idx in range(steps):
        new_branch_currents: List[np.ndarray] = []
        for idx, neuron in enumerate(neurons):
            presyn = [1.0 if idx == 0 and step_idx < steps // 2 else 0.0]
            for src, branch in incoming_map[idx]:
                prev_current = previous_branch_currents[src][branch]
                presyn.append(1.0 if prev_current > 0.0 else 0.0)
            result = neuron.step(presyn, external_current=7.0e-10 if idx == 0 else 2.0e-10)
            branch_currents_host = cp.asnumpy(result.branch_currents)
            new_branch_currents.append(branch_currents_host)
            if idx != 0 and branch_currents_host.any():
                propagation_hits[idx] += 1.0
            total_spikes += float(result.axon_spike)
        previous_branch_currents = new_branch_currents

    assert total_spikes > steps
    assert propagation_hits[1:].sum() > 0.0

    for neuron in neurons:
        input_weights = _to_numpy(neuron.input_weights)
        in_min, in_max = map(float, neuron.input_weight_bounds)
        assert np.all(input_weights >= in_min - 1e-12)
        assert np.all(input_weights <= in_max + 1e-12)

        output_weights = _to_numpy(neuron.output_weights)
        out_min, out_max = map(float, neuron.output_weight_bounds)
        assert np.all(output_weights >= out_min - 1e-12)
        assert np.all(output_weights <= out_max + 1e-12)

        dendritic = _to_numpy(neuron.postsynaptic_potentials)
        assert np.all(np.isfinite(dendritic))

    assert np.any(_to_numpy(neurons[10].postsynaptic_potentials) > 0.0)

if __name__ == "__main__":
    # To run these tests, you would typically use the pytest command.
    # However, you can run this script directly to execute the tests.
    import sys
    # The following will run all the tests in this file.
    # You can also pass arguments to pytest.main(), for example,
    # to run a specific test: pytest.main(['-k', 'test_large_cluster_signal_propagation'])
    sys.exit(pytest.main([__file__]))
