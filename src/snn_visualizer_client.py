import multiprocessing.shared_memory as shared_memory
import numpy as np
import struct
import time

class SNNVisualizer:
    def __init__(self, layer_structure, input_synapses, connections, shm_name="snn_visualization_shm"):
        """
        Initializes the SNNVisualizer and sets up the shared memory.
        Static data (network structure) is sent once upon initialization.

        :param layer_structure: List of integers, e.g., [10, 20, 10]
        :param input_synapses: Tuple of (target_indices, weights) for external inputs
        :param connections: Tuple of (source_indices, target_indices, weights) for inter-neuron connections
        :param shm_name: Name for the shared memory block
        """
        print("Initializing SNN Visualizer Client...")
        self.shm_name = shm_name
        self.shm = None
        self.version = 0

        # --- Unpack and store network structure ---
        self.layer_structure = layer_structure
        self.num_layers = len(layer_structure)
        self.total_neurons = sum(layer_structure)

        self.input_target_indices, self.input_weights = input_synapses
        self.num_input_synapses = len(self.input_target_indices)

        self.source_indices, self.target_indices, self.weights = connections
        self.num_connections = len(self.source_indices)

        # --- Calculate Buffer Size ---
        # Header: version (Q), num_layers (i), num_connections (i), num_input_synapses (i)
        self.header_format = '@Qiii'
        self.header_size = struct.calcsize(self.header_format)

        # Body: dynamic arrays of data
        body_format_str = (
            f'{self.num_layers}i'          # neurons_per_layer
            f'{self.total_neurons}f'      # neuron_states
            # Note: competition_values are removed as per user feedback
            f'{self.num_connections}i'    # source_indices
            f'{self.num_connections}i'    # target_indices
            f'{self.num_connections}f'    # weights
            f'{self.num_input_synapses}i' # input_target_indices
            f'{self.num_input_synapses}f' # input_weights
        )
        self.body_format = f'@{body_format_str}'
        self.body_size = struct.calcsize(self.body_format)
        self.buffer_size = self.header_size + self.body_size

        # --- Shared Memory Setup ---
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=self.buffer_size, name=self.shm_name)
            print(f"Created SHM '{self.shm_name}' with size {self.buffer_size} bytes.")
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            # Resize if existing buffer is smaller than required
            if self.shm.size < self.buffer_size:
                 # This path is complex; for now, we assume the user will manually clear SHM if structure changes.
                 # A robust implementation might involve recreating or signaling the C++ app to resize.
                raise ValueError(f"Existing SHM '{self.shm_name}' is too small. "
                                 f"Required: {self.buffer_size}, Found: {self.shm.size}. "
                                 "Please ensure the visualizer is closed and retry.")
            print(f"Attached to existing SHM '{self.shm_name}'.")

        print(f"Client ready. Sending: {self.num_layers} layers, {self.total_neurons} neurons, "
              f"{self.num_connections} connections, {self.num_input_synapses} inputs.")
        
        # Initial static data write
        self._write_static_data()


    def _write_static_data(self):
        """Packs and writes the non-changing structural data to the buffer."""
        # Pack structural data that doesn't change frame-to-frame
        # Note: neuron_states and weights will be dummy values initially
        dummy_neuron_states = np.zeros(self.total_neurons, dtype=np.float32)
        
        body_data = struct.pack(
            self.body_format,
            *self.layer_structure,
            *dummy_neuron_states,
            *self.source_indices,
            *self.target_indices,
            *self.weights,
            *self.input_target_indices,
            *self.input_weights
        )
        self.shm.buf[self.header_size:self.buffer_size] = body_data


    def update(self, neuron_states, weights=None, input_weights=None):
        """
        Updates the dynamic data (neuron states, weights) in shared memory.

        :param neuron_states: A numpy array of floats representing neuron activation.
        :param weights: (Optional) A numpy array of floats for inter-neuron synapse weights.
        :param input_weights: (Optional) A numpy array of floats for input synapse weights.
        """
        if self.shm is None:
            print("Warning: Shared memory not available. Update call ignored.")
            return

        # 1. Pack header
        header_data = struct.pack(self.header_format, self.version, self.num_layers, self.num_connections, self.num_input_synapses)
        
        # 2. Prepare data for packing
        current_weights = self.weights if weights is None else weights
        current_input_weights = self.input_weights if input_weights is None else input_weights

        # 3. Pack body
        body_data = struct.pack(
            self.body_format,
            *self.layer_structure,
            *neuron_states,
            *self.source_indices,
            *self.target_indices,
            *current_weights,
            *self.input_target_indices,
            *current_input_weights
        )

        # 4. Write to shared memory
        self.shm.buf[:self.header_size] = header_data
        self.shm.buf[self.header_size:self.buffer_size] = body_data

        self.version += 1

    def close(self):
        """Closes and unlinks the shared memory."""
        if self.shm is not None:
            self.shm.close()
            try:
                # Only the creator should unlink
                self.shm.unlink()
                print("Shared memory unlinked.")
            except FileNotFoundError:
                pass # It was already unlinked by another process
        self.shm = None

if __name__ == '__main__':
    # Example Usage and Test
    print("--- Running SNNVisualizer Module Test ---")
    
    # --- Configuration ---
    TEST_LAYER_STRUCTURE = [10, 20, 10]
    TEST_NUM_INPUTS = 5

    def generate_dummy_connections(layer_structure):
        import random
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
        import random
        input_target_indices, input_weights = [], []
        for _ in range(num_inputs):
            input_target_indices.append(random.randrange(first_layer_size))
            input_weights.append(random.random())
        return input_target_indices, input_weights

    # --- Data Preparation ---
    connections = generate_dummy_connections(TEST_LAYER_STRUCTURE)
    input_synapses = generate_dummy_input_synapses(TEST_NUM_INPUTS, TEST_LAYER_STRUCTURE[0])
    total_neurons = sum(TEST_LAYER_STRUCTURE)

    visualizer = None
    try:
        # --- Initialization ---
        visualizer = SNNVisualizer(
            layer_structure=TEST_LAYER_STRUCTURE,
            connections=connections,
            input_synapses=input_synapses
        )

        # --- Main Loop ---
        print("Starting data transmission loop...")
        while True:
            # Generate dynamic data
            neuron_states = np.random.rand(total_neurons).astype(np.float32)
            
            # Update visualizer
            visualizer.update(neuron_states)
            
            if visualizer.version % 120 == 0:
                print(f"Version: {visualizer.version}")

            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\nShutting down test...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if visualizer is not None:
            visualizer.close()
            print("Visualizer closed.")
