import cupy as cp
from .gpu_utils import GPUUtils

class SynapseOps:
    def __init__(self, gpu_utils: GPUUtils, N_post: int):
        self.N_post = N_post
        self.gpu = gpu_utils

        # GPU 메모리
        self.I_syn = cp.zeros(N_post, dtype=cp.float32)

        # Synapse CSR (예시)
        self.conn_idx = None
        self.conn_offset = None
        self.weights = None
        self.pre_spikes = None

        # 커널 로드
        self.gpu.load_kernel("synapse", "synapse_kernels.cu",
                             ["compute_synaptic_currents_kernel"])
        self.compute_currents = self.gpu.get_function("synapse", "compute_synaptic_currents_kernel")

    def step(self, block=(128,), grid=None):
        N = self.N_post
        if grid is None:
            grid = ((N + block[0] - 1) // block[0],)

        self.compute_currents(grid, block,
            (self.I_syn, self.pre_spikes, self.weights, self.conn_idx, self.conn_offset, self.N_post))
