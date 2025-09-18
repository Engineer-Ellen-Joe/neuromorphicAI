#include "common.h"
#include <cuda_runtime.h>

extern "C" {

// Compute synaptic currents
__global__ void compute_synaptic_currents_kernel(
  float* I_syn,
  const int* pre_spikes,
  const float* weights,
  const int* conn_idx,
  const int* conn_offset,
  int N_post
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_post) return;

  float I = 0.0f;
  // TODO: loop over incoming synapses for neuron i
  I_syn[i] = I;
}

// Short-term plasticity (STP) update (optional)
__global__ void update_stp_kernel(
  float* u, float* x,
  const int* pre_spikes,
  float U, float tau_d, float tau_f,
  int N_syn
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_syn) return;

  // TODO: implement Tsodyks-Markram STP model
}

} // extern "C"
