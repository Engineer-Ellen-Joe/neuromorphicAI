import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

class GpuSNN:
    """전체 GPU 기반의 SNN 모델을 관리하는 클래스"""
    def __init__(self, dt: float):
        self.dt = dt
        self._setup_morphology_and_connections()
        self._setup_synapses()
        self._initialize_gpu_arrays()
        self._load_cuda_kernels()

    def _setup_morphology_and_connections(self):
        """뉴런의 형태(morphology)와 구획 간 연결 구조를 정의"""
        self.n_neurons = 1
        self.n_soma = 1; self.type_soma = 0
        self.n_ais = 1; self.type_ais = 1
        self.n_basal = 10; self.type_basal = 2
        self.n_apical = 10; self.type_apical = 3
        self.n_tuft = 4; self.type_tuft = 4
        self.n_compartments = self.n_soma + self.n_ais + self.n_basal + self.n_apical + self.n_tuft
        self.soma_idx, self.ais_idx = 0, 1
        self.basal_indices = list(range(2, 2 + self.n_basal))
        self.apical_indices = list(range(2 + self.n_basal, 2 + self.n_basal + self.n_apical))
        self.tuft_indices = list(range(2 + self.n_basal + self.n_apical, self.n_compartments))
        
        # 구획 타입 배열 생성
        self.compartment_type = np.zeros(self.n_compartments, dtype=np.int32)
        self.compartment_type[self.soma_idx] = self.type_soma
        self.compartment_type[self.ais_idx] = self.type_ais
        self.compartment_type[self.basal_indices] = self.type_basal
        self.compartment_type[self.apical_indices] = self.type_apical
        self.compartment_type[self.tuft_indices] = self.type_tuft

        connections = [
            (self.soma_idx, self.ais_idx, 0.8),
            (self.basal_indices[0], self.soma_idx, 0.6),
            (self.apical_indices[0], self.soma_idx, 0.6) ]
        for i in range(self.n_apical - 1): connections.append((self.apical_indices[i], self.apical_indices[i+1], 0.3))
        connections.append((self.apical_indices[-1], self.tuft_indices[0], 0.4))
        self.connection_pairs, self.connection_g_axial = [], []
        for c1, c2, g in connections: self.connection_pairs.extend([c1, c2, c2, c1]); self.connection_g_axial.extend([g, g])
        self.n_connections = len(self.connection_g_axial)

    def _setup_synapses(self):
        self.n_synapses = 2
        self.synapse_post_compartment_map = [self.basal_indices[3], self.soma_idx]
        self.synapse_gmax = [0.002, 0.005]
        self.synapse_tau_decay = [3.0, 8.0]
        self.synapse_E = [0.0, -70.0]

    def _get_resting_state_gates(self, V_rest=-65.0):
        return 0.308, 0.596, 0.317

    def _initialize_gpu_arrays(self):
        V_rest, (m0, h0, n0) = -65.0, self._get_resting_state_gates()
        # 구획 배열
        self.d_compartment_type = cp.array(self.compartment_type, dtype=cp.int32)
        self.V = cp.full(self.n_compartments, V_rest, dtype=cp.float64)
        self.m, self.h, self.n = (cp.full(self.n_compartments, val, dtype=cp.float64) for val in (m0, h0, n0))
        self.I_axial = cp.zeros(self.n_compartments, dtype=cp.float64)
        self.I_synaptic = cp.zeros(self.n_compartments, dtype=cp.float64)
        self.I_ext = cp.zeros(self.n_compartments, dtype=cp.float64)
        self.spike_output = cp.zeros(self.n_compartments, dtype=cp.uint8)
        self.g_na = cp.full(self.n_compartments, 120.0, dtype=cp.float64); self.g_na[self.ais_idx] = 200.0
        self.g_k, self.g_leak, self.C_m = (cp.full(self.n_compartments, val, dtype=cp.float64) for val in (36.0, 0.3, 1.0))
        self.E_na, self.E_k, self.E_leak = (cp.full(self.n_compartments, val, dtype=cp.float64) for val in (50.0, -77.0, -54.4))
        self.d_connection_pairs = cp.array(self.connection_pairs, dtype=cp.int32)
        self.d_connection_g_axial = cp.array(self.connection_g_axial, dtype=cp.float64)
        # 시냅스 배열
        self.d_synapse_post_compartment_map = cp.array(self.synapse_post_compartment_map, dtype=cp.int32)
        self.d_synapse_gmax = cp.array(self.synapse_gmax, dtype=cp.float64)
        self.d_synapse_tau_decay = cp.array(self.synapse_tau_decay, dtype=cp.float64)
        self.d_synapse_E = cp.array(self.synapse_E, dtype=cp.float64)
        self.synapse_weight = cp.full(self.n_synapses, 1.0, dtype=cp.float64)
        self.synapse_conductance = cp.zeros(self.n_synapses, dtype=cp.float64)

    def _load_cuda_kernels(self):
        with open('neuron_kernels.cu', 'r', encoding='utf-8') as f:
            cuda_source = f.read()
        self.update_gates_kernel = cp.RawKernel(cuda_source, 'update_gates')
        self.update_voltage_kernel = cp.RawKernel(cuda_source, 'update_voltage')
        self.compute_axial_currents_kernel = cp.RawKernel(cuda_source, 'compute_axial_currents')
        self.compute_synaptic_currents_kernel = cp.RawKernel(cuda_source, 'compute_synaptic_currents')

    def trigger_synapse(self, synapse_index: int):
        gmax = self.d_synapse_gmax[synapse_index].get().item()
        self.synapse_conductance[synapse_index] += gmax

    def step(self):
        block_size = 256
        comp_grid = (self.n_compartments + block_size - 1) // block_size
        conn_grid = (self.n_connections + block_size - 1) // block_size
        syn_grid = (self.n_synapses + block_size - 1) // block_size
        self.I_axial.fill(0); self.I_synaptic.fill(0)
        self.compute_axial_currents_kernel((conn_grid,), (block_size,), (self.n_connections, self.d_connection_pairs, self.d_connection_g_axial, self.V, self.I_axial))
        self.compute_synaptic_currents_kernel((syn_grid,), (block_size,), (self.n_synapses, self.dt, self.d_synapse_post_compartment_map, self.d_synapse_gmax, self.d_synapse_tau_decay, self.d_synapse_E, self.synapse_weight, self.synapse_conductance, self.V, self.I_synaptic))
        self.update_gates_kernel((comp_grid,), (block_size,), (self.n_compartments, self.dt, self.V, self.m, self.h, self.n))
        self.update_voltage_kernel((comp_grid,), (block_size,), (self.n_compartments, self.dt, self.C_m, self.g_leak, self.E_leak, self.g_k, self.E_k, self.g_na, self.E_na, -50.0, -65.0, self.m, self.h, self.n, self.I_ext, self.I_axial, self.I_synaptic, self.V, self.spike_output, self.d_compartment_type, self.type_ais))

if __name__ == '__main__':
    sim_time_ms = 50.0
    dt_ms = 0.01
    n_steps = int(sim_time_ms / dt_ms)
    snn = GpuSNN(dt=dt_ms)
    
    # 기록할 구획: Soma와 AIS
    soma_v_history = np.zeros(n_steps)
    ais_v_history = np.zeros(n_steps)
    time_axis = np.arange(n_steps) * dt_ms

    print(f"\n--- {sim_time_ms}ms 시뮬레이션 시작 ---")
    # --- 시뮬레이션 루프 ---
    for i in range(n_steps):
        # 10ms ~ 40ms 동안 Soma에 외부 전류 주입
        if int(10 / dt_ms) <= i < int(40 / dt_ms):
            snn.I_ext[snn.soma_idx] = 10.0 # 10 uA/cm^2
        else:
            snn.I_ext[snn.soma_idx] = 0.0

        snn.step()
        soma_v_history[i] = snn.V[snn.soma_idx].get()
        ais_v_history[i] = snn.V[snn.ais_idx].get()

    print("--- 시뮬레이션 종료 ---")

    # --- 그래프 그리기 ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, ais_v_history, label='AIS Voltage')
    plt.plot(time_axis, soma_v_history, label='Soma Voltage', alpha=0.7)
    plt.title("Voltage of Soma and AIS over Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()