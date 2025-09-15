import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import heapq

class AxonManager:
    """스파이크 전파와 전도 지연(Axon Delay)을 관리하는 클래스"""
    def __init__(self, snn_instance):
        self.snn = snn_instance
        self.axon_connections = [] # 각 축삭 연결: {'from': neuron_idx, 'to': synapse_idx, 'delay': ms}
        self.spike_queue = [] # (arrival_time_ms, synapse_idx)를 저장하는 우선순위 큐 (heap)

    def add_connection(self, from_neuron_idx: int, to_synapse_idx: int, delay_ms: float):
        """
        축삭 연결을 추가
        현재는 단일 뉴런 모델이므로 from_neuron_idx는 항상 0
        """
        self.axon_connections.append({'from': from_neuron_idx, 'to': to_synapse_idx, 'delay': delay_ms})

    def process_spikes(self, current_time_ms: float, spike_indices: np.ndarray):
        """
        발생한 스파이크를 처리하여 큐에 추가.
        - spike_indices: AIS 구획에서 발생한 스파이크 인덱스 배열
        """
        for ais_idx in spike_indices:
            neuron_idx = 0 # self.snn.compartment_neuron_map[ais_idx]
            for conn in self.axon_connections:
                if conn['from'] == neuron_idx:
                    arrival_time = current_time_ms + conn['delay']
                    heapq.heappush(self.spike_queue, (arrival_time, conn['to'])) # 우선순위 큐에 추가하여 지연 시뮬레이션

    def check_and_trigger_synapses(self, current_time_ms: float):
        """
        현재 시간에 도착한 스파이크가 있는지 확인하고 시냅스 활성화.
        - spike_queue는 arrival_time 기준으로 정렬되어 있음
        """
        while self.spike_queue and self.spike_queue[0][0] <= current_time_ms:
            _, synapse_idx = heapq.heappop(self.spike_queue)
            self.snn.trigger_synapse(synapse_idx)
            print(f"시간 {current_time_ms:.2f}ms: Axon 전파 완료. 시냅스 {synapse_idx}번 활성화!")

class GpuSNN:
    """전체 GPU 기반의 SNN 모델을 관리하는 클래스"""
    def __init__(self, dt: float):
        self.dt = dt
        self._setup_morphology_and_connections()
        self._setup_synapses()
        self._initialize_gpu_arrays()
        self._load_cuda_kernels()

    def _setup_morphology_and_connections(self):
        """
        뉴런 구획 수와 연결 구조 설정
        구획 타입: soma, AIS, basal, apical, tuft
        """
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

        # 구획 타입 배열 초기화
        self.compartment_type = np.zeros(self.n_compartments, dtype=np.int32)
        self.compartment_type[self.soma_idx] = self.type_soma
        self.compartment_type[self.ais_idx] = self.type_ais
        self.compartment_type[self.basal_indices] = self.type_basal
        self.compartment_type[self.apical_indices] = self.type_apical
        self.compartment_type[self.tuft_indices] = self.type_tuft

        # 축삭 연결 (연결 쌍과 전도 g 값)
        connections = [
            (self.soma_idx, self.ais_idx, 0.8),
            (self.basal_indices[0], self.soma_idx, 0.6),
            (self.apical_indices[0], self.soma_idx, 0.6) ]
        
        # apical trunk 연결
        for i in range(self.n_apical - 1): connections.append((self.apical_indices[i], self.apical_indices[i+1], 0.3))
        connections.append((self.apical_indices[-1], self.tuft_indices[0], 0.4)) # apical -> tuft 연결

        self.connection_pairs, self.connection_g_axial = [], []
        for c1, c2, g in connections: # 양방향으로 pair 생성
            self.connection_pairs.extend([c1, c2, c2, c1])
            self.connection_g_axial.extend([g, g])
        self.n_connections = len(self.connection_g_axial)

    def _setup_synapses(self):
        """시냅스 설정: 타겟 구획, 최대 전도도, 감쇠 상수, reversal potential"""
        self.n_synapses = 2
        self.synapse_post_compartment_map = [self.basal_indices[3], self.soma_idx]
        self.synapse_gmax = [0.002, 0.005]
        self.synapse_tau_decay = [3.0, 8.0]
        self.synapse_E = [0.0, -70.0]

    def _get_resting_state_gates(self, V_rest=-65.0):
        """HH 모델의 휴지 상태에서 m,h,n 초기값 반환"""
        return 0.308, 0.596, 0.317

    def _initialize_gpu_arrays(self):
        """GPU 배열 초기화"""
        V_rest, (m0, h0, n0) = -65.0, self._get_resting_state_gates()
        self.d_compartment_type = cp.array(self.compartment_type, dtype=cp.int32)
        self.V = cp.full(self.n_compartments, V_rest, dtype=cp.float64)
        self.m, self.h, self.n = (cp.full(self.n_compartments, val, dtype=cp.float64) for val in (m0, h0, n0))

        # 전류 초기화
        self.I_axial = cp.zeros(self.n_compartments, dtype=cp.float64)
        self.I_synaptic = cp.zeros(self.n_compartments, dtype=cp.float64)
        self.I_ext = cp.zeros(self.n_compartments, dtype=cp.float64)

        # 스파이크 기록
        self.spike_output = cp.zeros(self.n_compartments, dtype=cp.uint8)

        # 채널 및 구획 전기적 특성
        self.g_na = cp.full(self.n_compartments, 120.0, dtype=cp.float64); self.g_na[self.ais_idx] = 200.0
        self.g_k, self.g_leak, self.C_m = (cp.full(self.n_compartments, val, dtype=cp.float64) for val in (36.0, 0.3, 1.0))
        self.E_na, self.E_k, self.E_leak = (cp.full(self.n_compartments, val, dtype=cp.float64) for val in (50.0, -77.0, -54.4))

        # 연결과 시냅스 GPU 복사
        self.d_connection_pairs = cp.array(self.connection_pairs, dtype=cp.int32)
        self.d_connection_g_axial = cp.array(self.connection_g_axial, dtype=cp.float64)
        self.d_synapse_post_compartment_map = cp.array(self.synapse_post_compartment_map, dtype=cp.int32)
        self.d_synapse_gmax = cp.array(self.synapse_gmax, dtype=cp.float64)
        self.d_synapse_tau_decay = cp.array(self.synapse_tau_decay, dtype=cp.float64)
        self.d_synapse_E = cp.array(self.synapse_E, dtype=cp.float64)

        # 시냅스 상태
        self.synapse_weight = cp.full(self.n_synapses, 1.0, dtype=cp.float64)
        self.synapse_conductance = cp.zeros(self.n_synapses, dtype=cp.float64)

    def _load_cuda_kernels(self):
        """CUDA 커널 로드 (RawKernel)"""
        with open('kernels/neuron_kernels.cu', 'r', encoding='utf-8') as f:
            cuda_source = f.read()
        self.update_gates_kernel = cp.RawKernel(cuda_source, 'update_gates')
        self.update_voltage_kernel = cp.RawKernel(cuda_source, 'update_voltage')
        self.compute_axial_currents_kernel = cp.RawKernel(cuda_source, 'compute_axial_currents')
        self.compute_synaptic_currents_kernel = cp.RawKernel(cuda_source, 'compute_synaptic_currents')

    def trigger_synapse(self, synapse_index: int):
        """시냅스를 활성화: conductance 증가"""
        gmax = self.d_synapse_gmax[synapse_index].get().item()
        self.synapse_conductance[synapse_index] += gmax

    def step(self):
        """1 시뮬레이션 시간 스텝 수행: 축삭/시냅스 전류 -> 게이트 업데이트 -> 전압 업데이트"""
        block_size = 256
        comp_grid = (self.n_compartments + block_size - 1) // block_size
        conn_grid = (self.n_connections + block_size - 1) // block_size
        syn_grid = (self.n_synapses + block_size - 1) // block_size

        # GPU 전류 초기화
        self.I_axial.fill(0); self.I_synaptic.fill(0)
        """
        축삭 전류 계산 (compute_axial_currents_kernel)
        시냅스 전류 계산 (compute_synaptic_currents_kernel)
        게이트 변수 업데이트 (update_gates_kernel)
        전압 업데이트 및 스파이크 검출 (update_voltage_kernel)
        """
        self.compute_axial_currents_kernel((conn_grid,), (block_size,), (self.n_connections, self.d_connection_pairs, self.d_connection_g_axial, self.V, self.I_axial))
        self.compute_synaptic_currents_kernel((syn_grid,), (block_size,), (self.n_synapses, self.dt, self.d_synapse_post_compartment_map, self.d_synapse_gmax, self.d_synapse_tau_decay, self.d_synapse_E, self.synapse_weight, self.synapse_conductance, self.V, self.I_synaptic))
        self.update_gates_kernel((comp_grid,), (block_size,), (self.n_compartments, self.dt, self.V, self.m, self.h, self.n))
        self.update_voltage_kernel((comp_grid,), (block_size,), (self.n_compartments, self.dt, self.C_m, self.g_leak, self.E_leak, self.g_k, self.E_k, self.g_na, self.E_na, -50.0, -65.0, self.m, self.h, self.n, self.I_ext, self.I_axial, self.I_synaptic, self.V, self.spike_output, self.d_compartment_type, self.type_ais))

if __name__ == '__main__':
    sim_time_ms = 50.0
    dt_ms = 0.01
    n_steps = int(sim_time_ms / dt_ms)
    
    snn = GpuSNN(dt=dt_ms)
    axon = AxonManager(snn)
    # AIS(뉴런 0) -> 시냅스 0으로 1.5ms 지연 연결
    axon.add_connection(from_neuron_idx=0, to_synapse_idx=0, delay_ms=1.5)
    
    # 기록용 배열
    soma_v_history = np.zeros(n_steps)
    ais_v_history = np.zeros(n_steps)
    target_syn_comp_v_history = np.zeros(n_steps)
    target_comp_idx = snn.synapse_post_compartment_map[0]
    time_axis = np.arange(n_steps) * dt_ms

    print(f"\n--- {sim_time_ms}ms 시뮬레이션 시작 ---")
    for i in range(n_steps):
        current_t = time_axis[i]
        # 10ms ~ 20ms 동안만 전류 주입
        if int(10 / dt_ms) <= i < int(20 / dt_ms):
            snn.I_ext[snn.soma_idx] = 10.0
        else:
            snn.I_ext[snn.soma_idx] = 0.0

        snn.step() # GPU 시뮬레이션 1 스텝
        
        # CPU에서 스파이크 처리
        spikes = snn.spike_output.get()
        spike_indices = np.where(spikes == 1)[0]
        if len(spike_indices) > 0:
            print(f"시간 {current_t:.2f}ms: AIS (구획 {spike_indices})에서 스파이크 발생!")
            axon.process_spikes(current_t, spike_indices)
        
        # 도착한 스파이크가 있으면 시냅스 활성화
        axon.check_and_trigger_synapses(current_t)

        # 구획 전압 기록
        soma_v_history[i] = snn.V[snn.soma_idx].get()
        ais_v_history[i] = snn.V[snn.ais_idx].get()
        target_syn_comp_v_history[i] = snn.V[target_comp_idx].get()

    print("--- 시뮬레이션 종료 ---")

    plt.figure(figsize=(12, 8))
    plt.plot(time_axis, ais_v_history, label=f'AIS (Comp {snn.ais_idx})', zorder=3)
    plt.plot(time_axis, soma_v_history, label=f'Soma (Comp {snn.soma_idx})', alpha=0.8, zorder=2)
    plt.plot(time_axis, target_syn_comp_v_history, label=f'Synapse Target (Comp {target_comp_idx})', linestyle='--', zorder=1)
    plt.title("Voltages over Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()
