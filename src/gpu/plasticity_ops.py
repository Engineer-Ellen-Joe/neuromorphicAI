import cupy as cp
from .gpu_utils import GPUUtils

class PlasticityOps:
    def __init__(self, gpu_utils: GPUUtils, N_syn: int):
        self.N_syn = N_syn
        self.gpu = gpu_utils

        # GPU 메모리
        self.weights = cp.zeros(N_syn, dtype=cp.float32)
        self.pre_spikes = cp.zeros(N_syn, dtype=cp.int32)
        self.post_spikes = cp.zeros(N_syn, dtype=cp.int32)

        # 커널 로드
        self.gpu.load_kernel("plasticity", "plasticity_kernels.cu",
                             ["stdp_update_kernel"])
        self.stdp_update = self.gpu.get_function("plasticity", "stdp_update_kernel")

    def step(self, A_plus, A_minus, tau_plus, tau_minus, block=(128,), grid=None):
        N = self.N_syn
        if grid is None:
            grid = ((N + block[0] - 1) // block[0],)

        self.stdp_update(grid, block,
            (self.weights, self.pre_spikes, self.post_spikes,
             cp.float32(A_plus), cp.float32(A_minus),
             cp.float32(tau_plus), cp.float32(tau_minus), self.N_syn))
