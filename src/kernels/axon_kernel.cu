#include "common.h"
#include <cuda_runtime.h>

extern "C" {

// Axon delay ring buffer update
__global__ void update_delay_ring_kernel(
  int* delay_buffer,
  int buffer_len,
  int t,
  const int* spikes,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // TODO: write spike to delay ring buffer
}

// Branch routing with safety factor (SF)
__global__ void branch_routing_kernel(
  const int* spikes,
  int* branch_spikes,
  const float* safety_factor,
  int N_branches
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_branches) return;

  // TODO: probabilistic routing using SF
}

} // extern "C"