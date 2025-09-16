#include "common.h"
#include <cuda_runtime.h>

extern "C" {

// Hodgkin-Huxley gating update
__global__ void update_gates_kernel(
  float* m, float* h, float* n,
  const float* v,
  const SimConstants consts,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // TODO: implement HH gating update (alpha/beta functions)
}

// Voltage update with ionic & synaptic currents
__global__ void update_voltage_kernel(
  float* v,
  const float* m, const float* h, const float* n,
  const float* I_syn,
  const SimConstants consts,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // TODO: implement membrane voltage update (Euler integration)
}

// Spike detection
__global__ void detect_spikes_kernel(
  const float* v,
  int* spikes,
  float threshold,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  spikes[i] = (v[i] >= threshold) ? 1 : 0;
}

} // extern "C"
