import cupy as cp
from .gpu_utils import GPUUtils

class AxonOps:
    def __init__(self, gpu_utils: GPUUtils, N: int, delay_len: int):
        self.N = N
        self.delay_len = delay_len
        self.gpu = gpu_utils

        # Ring buffer
        self.delay_buffer = cp.zeros((delay_len, N), dtype=cp.int32)

        # 커널 로드
        self.gpu.load_kernel("axon", "axon_kernels.cu",
                             ["update_delay_ring_kernel", "branch_routing_kernel"])
        self.update_delay = self.gpu.get_function("axon", "update_delay_ring_kernel")
        self.branch_routing = self.gpu.get_function("axon", "branch_routing_kernel")

    def step_delay(self, spikes, t, block=(128,), grid=None):
        N = self.N
        if grid is None:
            grid = ((N + block[0] - 1) // block[0],)

        self.update_delay(grid, block,
            (self.delay_buffer, self.delay_len, t, spikes, self.N))
