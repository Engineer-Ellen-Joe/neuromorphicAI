#include "common.h"
#include <cuda_runtime.h>

extern "C" {

// HH gating update
__global__ void update_gates_kernel(
  float* m, float* h, float* n,
  const float* v,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // 예시: constant memory 접근
  float dt = d_consts.dt;
  float gNa = d_consts.g_na;

  // TODO: alpha/beta 계산하고 m,h,n 업데이트
}

// Voltage update
__global__ void update_voltage_kernel(
  float* v,
  const float* m, const float* h, const float* n,
  const float* I_syn,
  int N
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float Cm = d_consts.Cm;
  float dt = d_consts.dt;

  // TODO: 막전압 업데이트
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
