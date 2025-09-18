import cupy as cp
import numpy as np
from .gpu_utils import GPUUtils

# numpy dtype으로 CUDA struct 정의
SimConstants_dtype = np.dtype([
    ('dt', np.float32),
    ('g_na', np.float32),
    ('g_k', np.float32),
    ('g_leak', np.float32),
    ('E_na', np.float32),
    ('E_k', np.float32),
    ('E_leak', np.float32),
    ('Cm', np.float32),
])

class NeuronOps:
    def __init__(self, gpu_utils: GPUUtils, N: int, config):
        self.N = N
        self.gpu = gpu_utils

        # GPU 메모리
        self.v = cp.zeros(N, dtype=cp.float32)
        self.m = cp.zeros(N, dtype=cp.float32)
        self.h = cp.zeros(N, dtype=cp.float32)
        self.n = cp.zeros(N, dtype=cp.float32)
        self.spikes = cp.zeros(N, dtype=cp.int32)
        self.I_syn = cp.zeros(N, dtype=cp.float32)

        # 커널 로드
        module = self.gpu.load_kernel(
            "neuron", "neuron_kernel.cu",
            ["update_gates_kernel", "update_voltage_kernel", "detect_spikes_kernel"]
        )
        self.update_gates = module.get_function("update_gates_kernel")
        self.update_voltage = module.get_function("update_voltage_kernel")
        self.detect_spikes = module.get_function("detect_spikes_kernel")

        # 시뮬레이션 상수 업로드
        self.upload_constants(module, config)

    def upload_constants(self, module, config):
        """SimConstants struct를 GPU constant memory에 업로드"""
        consts = np.zeros((), dtype=SimConstants_dtype)
        consts['dt'] = config.dt
        consts['g_na'] = config.g_na
        consts['g_k'] = config.g_k
        consts['g_leak'] = config.g_leak
        consts['E_na'] = config.E_na
        consts['E_k'] = config.E_k
        consts['E_leak'] = config.E_leak
        consts['Cm'] = config.Cm

        memptr = module.get_global("d_consts")
        # NumPy 배열 객체가 아닌, 데이터의 실제 메모리 포인터를 전달해야 합니다.
        memptr.copy_from_host(consts.ctypes.data, consts.nbytes)


    def step(self, block=(128,), grid=None):
        N = self.N
        if grid is None:
            grid = ((N + block[0] - 1) // block[0],)

        self.update_gates(grid, block, (self.m, self.h, self.n, self.v, N))
        self.update_voltage(grid, block, (self.v, self.m, self.h, self.n, self.I_syn, N))
        self.detect_spikes(grid, block, (self.v, self.spikes, cp.float32(-50.0), N))
