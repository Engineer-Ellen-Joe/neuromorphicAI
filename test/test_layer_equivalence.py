""" Compares the behavior of PyramidalNeuron and PyramidalLayer.

This script performs an equivalence test to ensure that a PyramidalLayer
with a single neuron behaves identically to a standalone PyramidalNeuron
given the same parameters and inputs. It runs a simulation for both, then
compares their state variables numerically and visually.
"""

import sys
import os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Add project root to path to import source modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_neuron import PyramidalNeuron
from src.pyramidal_layer import PyramidalLayer

# --- Configuration ---
GPU_AVAILABLE = cp.cuda.is_available()
if GPU_AVAILABLE:
    try:
        plt.rcParams['font.family'] = 'Noto Sans KR'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("Noto Sans KR font not found. Using default font.")

def _to_numpy(array):
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA device required for this test")
def test_layer_neuron_equivalence():
    """
    Tests if a single-neuron PyramidalLayer produces identical output to 
    a PyramidalNeuron over a simulation period.
    """
    # --- 1. Identical Setup ---
    print("--- Setting up PyramidalNeuron and PyramidalLayer(num_neurons=1) ---")
    
    # Shared parameters
    num_afferents = 10
    num_branches = 5
    dt = 1e-4
    random_seed = 42
    
    # Use numpy for initial weights to ensure they are identical
    np.random.seed(random_seed)
    initial_input_weights = np.random.uniform(0.4, 0.6, size=(num_afferents,)).astype(np.float64)
    initial_output_weights = np.random.uniform(0.6, 0.8, size=(num_branches,)).astype(np.float64)
    safety_factors = np.random.uniform(0.8, 1.2, size=(num_branches,)).astype(np.float64)

    common_params = {
        'num_afferents': num_afferents,
        'num_branches': num_branches,
        'dt': dt,
        'random_state': random_seed,
        'initial_input_weights': initial_input_weights,
        'initial_output_weights': initial_output_weights,
        'safety_factors': safety_factors,
        'input_learning_rate': 0.01,
        'output_learning_rate': 0.02,
    }

    # Instantiate the single neuron
    neuron = PyramidalNeuron(**common_params)

    # Instantiate the layer with one neuron
    layer = PyramidalLayer(
        num_neurons=1,
        **{
            k: v if k not in ['initial_input_weights', 'initial_output_weights', 'safety_factors'] 
            else [v] # Wrap weights/factors in another list for the layer
            for k, v in common_params.items()
        }
    )

    # --- 2. Simulation ---
    print("--- Running parallel simulation ---")
    duration = 0.5  # seconds
    num_steps = int(duration / dt)
    time = np.arange(num_steps) * dt

    # Generate a complex input spike train
    np.random.seed(random_seed + 1)
    presynaptic_spikes_train = cp.array(np.random.rand(num_steps, num_afferents) > 0.9, dtype=cp.float64)
    external_current_train = cp.array(np.sin(np.linspace(0, 20 * np.pi, num_steps)) * 5e-10, dtype=cp.float64)

    # History storage
    history = {
        'neuron': {'potential': [], 'input_w': [], 'output_w': []},
        'layer': {'potential': [], 'input_w': [], 'output_w': []}
    }

    for i in range(num_steps):
        # Get inputs for the current step
        current_spikes = presynaptic_spikes_train[i]
        current_ext = external_current_train[i]

        # Step both models
        neuron_res = neuron.step(presynaptic_spikes=current_spikes, external_current=float(current_ext))
        layer_res = layer.step(presynaptic_spikes=current_spikes, external_currents=cp.array([current_ext]))

        # --- 3. Numerical Verification (at each step) ---
        assert np.allclose(neuron_res.soma_potential, _to_numpy(layer_res.soma_potentials)[0]), f"Step {i}: Soma potential mismatch"
        assert np.allclose(neuron_res.axon_spike, _to_numpy(layer_res.axon_spikes)[0]), f"Step {i}: Axon spike mismatch"
        assert cp.allclose(neuron.input_weights, layer.input_weights[0]), f"Step {i}: Input weights mismatch"
        assert cp.allclose(neuron.output_weights, layer.output_weights[0]), f"Step {i}: Output weights mismatch"
        assert cp.allclose(neuron.pre_trace, layer.pre_trace[0]), f"Step {i}: Pre-trace mismatch"
        assert cp.allclose(neuron.post_trace, layer.post_trace[0]), f"Step {i}: Post-trace mismatch"
        assert cp.allclose(neuron.bcm_theta, layer.bcm_theta[0]), f"Step {i}: BCM theta mismatch"

        # Store history for plotting
        history['neuron']['potential'].append(neuron_res.soma_potential)
        history['layer']['potential'].append(_to_numpy(layer_res.soma_potentials)[0])
        if i % 100 == 0:
            history['neuron']['input_w'].append(_to_numpy(neuron.input_weights).copy())
            history['layer']['input_w'].append(_to_numpy(layer.input_weights[0]).copy())
            history['neuron']['output_w'].append(_to_numpy(neuron.output_weights).copy())
            history['layer']['output_w'].append(_to_numpy(layer.output_weights[0]).copy())

    print("--- Numerical equivalence test PASSED ---")

    # --- 4. Visual Verification ---
    print("--- Generating comparison plots ---")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('PyramidalNeuron vs. PyramidalLayer(1) Equivalence Test', fontsize=16)

    # Plot 1: Soma Potential
    axes[0].plot(time * 1000, np.array(history['neuron']['potential']) * 1000, 'b-', label='Neuron (Original)')
    axes[0].plot(time * 1000, np.array(history['layer']['potential']) * 1000, 'r--', label='Layer[0] (New)', alpha=0.7)
    axes[0].set_ylabel('Soma Potential (mV)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Soma Potential Over Time')

    # Plot 2: Input Weights
    neuron_input_w_hist = np.array(history['neuron']['input_w'])
    layer_input_w_hist = np.array(history['layer']['input_w'])
    for j in range(num_afferents):
        axes[1].plot(neuron_input_w_hist[:, j], color='blue', alpha=0.5, linestyle='-')
        axes[1].plot(layer_input_w_hist[:, j], color='red', alpha=0.5, linestyle='--')
    axes[1].set_ylabel('Input Weights')
    axes[1].set_title('Input Weight Dynamics (Blue=Original, Red=New)')
    axes[1].grid(True)

    # Plot 3: Output Weights
    neuron_output_w_hist = np.array(history['neuron']['output_w'])
    layer_output_w_hist = np.array(history['layer']['output_w'])
    for j in range(num_branches):
        axes[2].plot(neuron_output_w_hist[:, j], color='blue', alpha=0.5, linestyle='-')
        axes[2].plot(layer_output_w_hist[:, j], color='red', alpha=0.5, linestyle='--')
    axes[2].set_xlabel('Time (steps, sampled)')
    axes[2].set_ylabel('Output Weights')
    axes[2].set_title('Output Weight Dynamics (BCM)')
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = "test_layer_equivalence.png"
    plt.savefig(output_filename)
    print(f"--- Comparison plot saved to {os.path.abspath(output_filename)} ---")
    plt.close(fig) # Close the figure to free up memory

if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("Equivalence Test: CUDA device not found. Skipping test.")
    else:
        pytest.main(["-s", "-v", __file__])
