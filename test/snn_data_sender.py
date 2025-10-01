import multiprocessing.shared_memory as shared_memory
import cupy as cp
import numpy as np
import time
import struct
import random

# --- Configuration ---
LAYER_STRUCTURE = [10, 20, 10]
NUM_INPUTS = 5 # Number of external input signals
SHM_NAME = "snn_visualization_shm"

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
    print("SNN Data Sender starting with FINAL structure...")
    
    # --- Prepare Data Structure ---
    NUM_LAYERS = len(LAYER_STRUCTURE)
    TOTAL_NEURONS = sum(LAYER_STRUCTURE)
    
    source_indices, target_indices, weights = generate_dummy_connections(LAYER_STRUCTURE)
    NUM_CONNECTIONS = len(source_indices)

    input_target_indices, input_weights = generate_dummy_input_synapses(NUM_INPUTS, LAYER_STRUCTURE[0])
    NUM_INPUT_SYNAPSES = len(input_target_indices)

    # --- Calculate Buffer Size ---
    # Header: version, num_layers, num_connections, num_input_synapses
    header_format = '@Qiii'
    header_size = struct.calcsize(header_format)

    # Body: arrays of data
    body_format_str = (
        f'{NUM_LAYERS}i'          # neurons_per_layer
        f'{TOTAL_NEURONS}f'      # neuron_states
        f'{TOTAL_NEURONS}f'      # competition_values
        f'{NUM_CONNECTIONS}i'    # source_indices
        f'{NUM_CONNECTIONS}i'    # target_indices
        f'{NUM_CONNECTIONS}f'    # weights
        f'{NUM_INPUT_SYNAPSES}i' # input_target_indices
        f'{NUM_INPUT_SYNAPSES}f' # input_weights
    )
    body_format = f'@{body_format_str}'
    body_size = struct.calcsize(body_format)
    BUFFER_SIZE = header_size + body_size
    
    shm = None
    try:
        # --- Shared Memory Setup ---
        try:
            shm = shared_memory.SharedMemory(create=True, size=BUFFER_SIZE, name=SHM_NAME)
            print(f"Created SHM '{SHM_NAME}' with size {BUFFER_SIZE} bytes.")
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            print(f"Attached to existing SHM '{SHM_NAME}'.")

        print(f"Sending: {NUM_LAYERS} layers, {TOTAL_NEURONS} neurons, {NUM_CONNECTIONS} connections, {NUM_INPUT_SYNAPSES} inputs.")

        version = 0
        while True:
            # 1. Generate dummy data
            cpu_neuron_states = np.random.rand(TOTAL_NEURONS).astype(np.float32)
            cpu_competition_values = np.random.rand(TOTAL_NEURONS).astype(np.float32)
            # In a real scenario, weights would also be updated

            # 2. Pack header
            header_data = struct.pack(header_format, version, NUM_LAYERS, NUM_CONNECTIONS, NUM_INPUT_SYNAPSES)

            # 3. Pack body
            body_data = struct.pack(
                body_format,
                *LAYER_STRUCTURE,
                *cpu_neuron_states,
                *cpu_competition_values,
                *source_indices,
                *target_indices,
                *weights,
                *input_target_indices,
                *input_weights
            )

            # 4. Write to shared memory
            shm.buf[:header_size] = header_data
            shm.buf[header_size:BUFFER_SIZE] = body_data

            if version % 120 == 0:
                print(f"Version: {version}")

            version += 1
            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if shm is not None:
            shm.close()
            shm.unlink()
            print("Shared memory unlinked.")

if __name__ == "__main__":
    main()
