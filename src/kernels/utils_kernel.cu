#include <curand_kernel.h>

extern "C" {

// RNG initialization
__global__ void init_rng_kernel(
  curandState* rng_states,
  unsigned long seed,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  curand_init(seed, i, 0, &rng_states[i]);
}

// Random number generation (uniform)
__device__ float rand_uniform(curandState* state) {
  return curand_uniform(state);
}

// Prefix-sum, scatter/gather (placeholder)
__global__ void prefix_sum_kernel(int* data, int N) {
  // TODO: implement scan
}

} // extern "C"
