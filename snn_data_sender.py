import multiprocessing.shared_memory as shared_memory
import cupy as cp
import numpy as np
import time
import struct

# --- Configuration ---
NUM_NEURONS = 100
SHM_NAME = "snn_visualization_shm"
# 8 bytes for version counter (unsigned long long) + 100 neurons * 4 bytes/float
BUFFER_SIZE = 8 + NUM_NEURONS * 4 

def main():
    print("SNN Data Sender starting...")
    shm = None
    try:
        # Create the shared memory block
        try:
            shm = shared_memory.SharedMemory(create=True, size=BUFFER_SIZE, name=SHM_NAME)
            print(f"Created shared memory block '{SHM_NAME}' with size {BUFFER_SIZE} bytes.")
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            print(f"Attached to existing shared memory block '{SHM_NAME}'.")

        # CuPy setup
        device = cp.cuda.Device(0)
        device.use()
        gpu_properties = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"Using GPU: {gpu_properties['name'].decode()}")

        version = 0
        while True:
            # 1. Generate dummy data on the GPU
            gpu_data = cp.random.rand(NUM_NEURONS, dtype=cp.float32)

            # 2. Copy data from GPU to CPU
            cpu_data = gpu_data.get() # Returns a numpy array

            # 3. Write to shared memory
            # Pack version counter (unsigned long long)
            shm.buf[0:8] = struct.pack('Q', version)
            # Pack neuron data (100 floats)
            shm.buf[8:] = cpu_data.tobytes()

            if version % 120 == 0: # Print status every 120 frames
                print(f"Version: {version}, First neuron value: {cpu_data[0]:.4f}")

            version += 1
            time.sleep(1/60) # ~60 FPS

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if shm is not None:
            shm.close()
            shm.unlink() # Free the shared memory block
            print("Shared memory unlinked.")

if __name__ == "__main__":
    main()
