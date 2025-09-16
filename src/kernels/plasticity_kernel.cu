#include "common.h"
#include <cuda_runtime.h>

extern "C" {

// STDP weight update
__global__ void stdp_update_kernel(
  float* weights,
  const int* pre_spikes,
  const int* post_spikes,
  float A_plus, float A_minus,
  float tau_plus, float tau_minus,
  int N_syn
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_syn) return;

  // TODO: implement exponential STDP
}

} // extern "C"
