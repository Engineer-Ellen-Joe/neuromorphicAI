import cupy as cp
from .gpu_utils import GPUUtils

class NeuronOps:
    def __init__(self, gpu_utils: GPUUtils, N: int):
        self.N = N
        self.gpu = gpu_utils

        # GPU 메모리 할당
        self.v = cp.zeros(N, dtype=cp.float32)
        self.m = cp.zeros(N, dtype=cp.float32)
        self.h = cp.zeros(N, dtype=cp.float32)
        self.n = cp.zeros(N, dtype=cp.float32)
        self.spikes = cp.zeros(N, dtype=cp.int32)
        self.I_syn = cp.zeros(N, dtype=cp.float32)

        # 커널 로드
        self.gpu.load_kernel("neuron", "neuron_kernel.cu",
                             ["update_gates_kernel", "update_voltage_kernel", "detect_spikes_kernel"])

        self.update_gates = self.gpu.get_function("neuron", "update_gates_kernel")
        self.update_voltage = self.gpu.get_function("neuron", "update_voltage_kernel")
        self.detect_spikes = self.gpu.get_function("neuron", "detect_spikes_kernel")

    def step(self, sim_consts, block=(128,), grid=None):
        N = self.N
        if grid is None:
            grid = ((N + block[0] - 1) // block[0],)

        # TODO: sim_consts를 GPU 상수 메모리로 전달하거나 struct 변환 필요
        self.update_gates(grid, block, (self.m, self.h, self.n, self.v, sim_consts, N))
        self.update_voltage(grid, block, (self.v, self.m, self.h, self.n, self.I_syn, sim_consts, N))
        self.detect_spikes(grid, block, (self.v, self.spikes, cp.float32(-50.0), N))
