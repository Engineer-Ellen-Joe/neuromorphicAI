from __future__ import annotations
import cupy as cp
from dataclasses import dataclass
from typing import Optional

@dataclass
class SomaParams:
  """상세 뉴런 집단의 세포체(soma) 구획을 위한 파라미터들"""
  C_m: float = 1.6  # 막전위 용량(Membrane capacitance) (uF/cm^2)
  g_leak: float = 0.3
  E_leak: float = -65.0
  # K channel params
  g_k: float = 36.0
  E_k: float = -77.0
  # Na channel params
  g_na: float = 120.0
  E_na: float = 50.0
  # 발화(spike)
  spike_threshold: float = 0.0
  V_reset: float = -65.0

class DetailedNeuronPopulation:
  """
  생물리학적으로 상세한, 다중 구획(multi-compartment) 뉴런 집단.
  안정성을 위해 배정밀도(double precision)를 사용함.
  !! 모든 채널 활성화 상태 !!
  """
  def __init__(self, N: int, dt: float, params: Optional[SomaParams] = None, name: str = "detailed_neurons"):
    self.N = N
    self.dt = dt
    self.name = name
    self.p = params or SomaParams()

    self.V_soma = cp.full(N, self.p.E_leak, dtype=cp.float64)
    self.n = cp.zeros(N, dtype=cp.float64)  # K+ channel activation
    self.m = cp.zeros(N, dtype=cp.float64)  # Na+ channel activation
    self.h = cp.ones(N, dtype=cp.float64)   # Na+ channel inactivation
    self.spike = cp.zeros(N, dtype=cp.uint8)

    with open('neuron_kernels.cu', 'r', encoding='utf-8') as f:
      cuda_source = f.read()

    ####################################################################################################
    # CUDA 커널 1
    # 게이트 변수(m, h, n) 업데이트
    ####################################################################################################
    self._update_gates_kernel = cp.RawKernel(cuda_source, 'update_gates')

    ####################################################################################################
    # CUDA: 커널 2
    # 막 전위(V) 업데이트
    ####################################################################################################
    self._update_voltage_kernel = cp.RawKernel(cuda_source, 'update_voltage')

  def step(self, Iext: Optional[cp.ndarray] = None):
    threads_per_block = 256
    blocks_per_grid = (self.N + threads_per_block - 1) // threads_per_block

    # 게이트 업데이트
    self._update_gates_kernel(
      (blocks_per_grid,), (threads_per_block,),
      (self.N, self.dt, self.V_soma, self.m, self.h, self.n)
    )

    # 외부 전류 없으면 0으로 채움
    if Iext is None:
      Iext = cp.zeros(self.N, dtype=cp.float64)

    # 막 전위 업데이트
    self._update_voltage_kernel(
      (blocks_per_grid,), (threads_per_block,),
        (
          self.N, self.dt, self.p.C_m,
          self.p.g_leak, self.p.E_leak,
          self.p.g_k, self.p.E_k,
          self.p.g_na, self.p.E_na,
          self.p.spike_threshold, self.p.V_reset,
          self.m, self.h, self.n,
          Iext,
          self.V_soma,
          self.spike
        )
    )

if __name__ == "__main__":
  import numpy as np
  import matplotlib.pyplot as plt

  # Simulation parameters
  N_NEURONS = 1
  SIM_TIME_MS = 200.0
  DT_MS = 0.01
  I_STIM_NA = 10.0

  print("--- Detailed Neuron Model Test (All Channels, Double Precision) ---")
    
  neurons = DetailedNeuronPopulation(N=N_NEURONS, dt=DT_MS)

  n_steps = int(SIM_TIME_MS / DT_MS)
  v_history = np.zeros((n_steps, N_NEURONS), dtype=np.float64)
    
  I_ext = cp.zeros(N_NEURONS, dtype=cp.float64)
  start_stim_ms = 50
  end_stim_ms = 52
  start_step = int(start_stim_ms / DT_MS)
  end_step = int(end_stim_ms / DT_MS)
    
  print(f"Applying {I_STIM_NA}uA/cm^2 current from {start_stim_ms}ms to {end_stim_ms}ms.")

  for i in range(n_steps):
    if start_step <= i < end_step:
      I_ext[0] = I_STIM_NA
    else:
      I_ext[0] = 0.0

    neurons.step(I_ext)
    v_history[i, :] = neurons.V_soma.get()

  print("Simulation finished.")

  # Plot the results
  time_axis = np.arange(0, SIM_TIME_MS, DT_MS)
  plt.figure(figsize=(12, 6))
  plt.plot(time_axis, v_history[:, 0])
  plt.title("Membrane Potential of a Detailed Neuron (All Channels)")
  plt.xlabel("Time (ms)")
  plt.ylabel("Voltage (mV)")
  plt.grid(True)
  plt.show()