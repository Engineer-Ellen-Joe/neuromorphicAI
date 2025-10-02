import numpy as np
import time
import random
import sys
import os

# Add src directory to Python path to import the new module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from snn_visualizer_client import SNNVisualizer

# --- Configuration ---
LAYER_STRUCTURE = [10, 20, 10]
NUM_INPUTS = 5 # Number of external input signals

def generate_dummy_connections(layer_structure):
    source_indices, target_indices, weights = [], [], []
    neuron_idx_offset = 0
    for i in range(len(layer_structure) - 1):
        layer_from_size = layer_structure[i]
        layer_to_size = layer_structure[i+1]
        for j in range(layer_from_size):
            num_connections = random.randint(1, 2)
            for _ in range(num_connections):
                source_idx = neuron_idx_offset + j
                target_idx = neuron_idx_offset + layer_from_size + random.randrange(layer_to_size)
                source_indices.append(source_idx)
                target_indices.append(target_idx)
                weights.append(random.random())
        neuron_idx_offset += layer_from_size
    return source_indices, target_indices, weights

def generate_dummy_input_synapses(num_inputs, first_layer_size):
    input_target_indices, input_weights = [], []
    for _ in range(num_inputs):
        input_target_indices.append(random.randrange(first_layer_size))
        input_weights.append(random.random())
    return input_target_indices, input_weights

def main():
    """
    This script now acts as a simple test case for the SNNVisualizer module.
    """
    print("--- Starting Test Script for SNNVisualizer Module ---")

    # --- 1. Prepare Network Structure Data ---
    connections = generate_dummy_connections(LAYER_STRUCTURE)
    input_synapses = generate_dummy_input_synapses(NUM_INPUTS, LAYER_STRUCTURE[0])
    total_neurons = sum(LAYER_STRUCTURE)

    visualizer = None
    try:
        # --- 2. Initialize the Visualizer ---
        # This will create the shared memory and send the network structure.
        visualizer = SNNVisualizer(
            layer_structure=LAYER_STRUCTURE,
            connections=connections,
            input_synapses=input_synapses
        )

        # --- 3. Run the Main Loop ---
        # In a real application, this loop would be part of the SNN simulation.
        print("Starting data transmission loop...")
        while True:
            # Generate dynamic data (e.g., from the SNN simulation)
            neuron_states = np.random.rand(total_neurons).astype(np.float32)
            
            # Update the visualizer with the new neuron states
            visualizer.update(neuron_states)
            
            if visualizer.version % 120 == 0:
                print(f"Version: {visualizer.version}")

            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- 4. Clean up ---
        if visualizer is not None:
            visualizer.close()
            print("Visualizer closed.")

if __name__ == "__main__":
    main()