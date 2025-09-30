"""
2-Layer SNN의 계층적 학습(Hierarchical Learning)을 검증합니다.

이 스크립트는 새로운 pybind11 기반 아키텍처를 사용하여 Vulkan-CUDA 메모리 공유를 테스트합니다.
C++ 확장 모듈(vkdbg)이 Vulkan 버퍼를 생성하고, Python 브릿지(gpu_vulkan_bridge)가
이를 CuPy 배열로 매핑하여 실시간으로 데이터를 공유합니다.
"""

import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

# --- 새로운 아키텍처에 맞게 경로 및 임포트 수정 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

# --- DLL 로드 문제 해결을 위한 경로 추가 ---
try:
    os.add_dll_directory("C:/VulkanSDK/1.4.313.1/Bin")
    os.add_dll_directory("C:/msys64/ucrt64/bin")
except AttributeError:
    # os.add_dll_directory is Python 3.8+ only
    pass

from src.pyramidal_layer import PyramidalLayer, DTYPE

# Vulkan 초기화를 위한 LveWindow, LveDevice 임포트
# 참고: 이 클래스들은 원래 src/vulkan/src/에 정의되어 있어야 합니다.
# 여기서는 해당 클래스가 Python에서 사용 가능하다고 가정합니다.
# 지금은 개념 증명을 위해 LveDevice가 있다고 가정하고 진행합니다.
from vulkan_setup import LveWindow, LveDevice

import vkdbg
import python.gpu_vulkan_bridge as bridge

# --- 공유 버퍼 관리를 위한 헬퍼 클래스 ---
class SharedBufferManager:
    def __init__(self):
        self.buffers = {}
        self.bridge_initialized = False

    def _initialize_bridge(self, vulkan_uuid):
        if not self.bridge_initialized:
            print(f"[Manager] Initializing bridge for Vulkan device UUID: {vulkan_uuid}")
            bridge.init_cuda_for_vulkan_device(vulkan_uuid)
            self.bridge_initialized = True

    def update_buffer(self, name, data):
        if name not in self.buffers:
            print(f"[Manager] Creating new shared buffer: {name}")
            info = vkdbg.create_exportable_buffer(data.nbytes)
            
            # Initialize bridge on first buffer creation
            self._initialize_bridge(info.uuid)

            arr, ext_mem = bridge.import_and_map(info)
            
            shared_array = arr.view(dtype=data.dtype).reshape(data.shape)

            self.buffers[name] = {
                'info': info,
                'array': shared_array,
                'ext_mem': ext_mem
            }
        
        self.buffers[name]['array'][:] = data

    def cleanup(self):
        print("[Manager] Cleaning up all shared buffers...")
        for name, buf_data in self.buffers.items():
            bridge.destroy_mapped(buf_data['ext_mem'])
            vkdbg.destroy_exportable_buffer(buf_data['info'].internal_id)
        self.buffers = {}

# --- 기본 설정 ---
GPU_AVAILABLE = cp.cuda.is_available()

# (기존 verify_and_visualize_layer1, define_patterns 함수들은 변경 없음 - 그대로 사용)
def verify_and_visualize_layer1(layer, patterns, dt, base_current, inhibition_current):
    """학습된 1계층의 역할 분담 결과를 검증하고 시각화합니다."""
    print("\n--- 중간 검증: Layer 1 역할 분담 확인 ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    
    spike_counts = cp.zeros((2, layer.num_neurons), dtype=cp.int32)
    test_patterns = [pA_spikes, pB_spikes]
    
    base_current_gpu = cp.full(layer.num_neurons, 0.2, dtype=DTYPE)

    original_lr = layer.input_learning_rate
    layer.input_learning_rate = 0.0

    for i, pattern_spikes in enumerate(test_patterns):
        layer.reset_state()
        for _ in range(5):
            for step in range(p_steps):
                presynaptic_spikes = cp.zeros(layer.num_afferents, dtype=DTYPE)
                for t, afferent_idx in pattern_spikes:
                    if step == t:
                        presynaptic_spikes[afferent_idx] = 1.0
                
                result = layer.step_competitive(
                    presynaptic_spikes,
                    external_currents=base_current_gpu,
                    inhibition_current=inhibition_current
                )
                spike_counts[i] += (result.axon_spikes > 0).astype(cp.int32)

    layer.input_learning_rate = original_lr

    spike_counts_cpu = spike_counts.get()
    neuron_indices = np.arange(layer.num_neurons)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.4
    
    colors_A = ['blue'] * (layer.num_neurons // 2) + ['lightgray'] * (layer.num_neurons - layer.num_neurons // 2)
    colors_B = ['lightgray'] * (layer.num_neurons // 2) + ['red'] * (layer.num_neurons - layer.num_neurons // 2)

    ax.bar(neuron_indices - bar_width/2, spike_counts_cpu[0], bar_width, label='Response to Pattern A', color=colors_A)
    ax.bar(neuron_indices + bar_width/2, spike_counts_cpu[1], bar_width, label='Response to Pattern B', color=colors_B)

    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Total Spikes (5 trials)')
    ax.set_title('Layer 1 Neuron Specialization Verification')
    ax.set_xticks(neuron_indices)
    ax.legend()
    fig.tight_layout()
    
    save_path = os.path.join(os.path.dirname(__file__), 'l1_specialization_check.png')
    plt.savefig(save_path)
    print(f"  - 검증 그래프 저장 완료: {save_path}")
    plt.close(fig)

def define_patterns(num_afferents, dt):
    """학습에 사용할 두 개의 시공간 패턴을 정의합니다."""
    pattern_A_afferents = np.arange(0, num_afferents // 2)
    pattern_B_afferents = np.arange(num_afferents // 2, num_afferents)
    pattern_interval_ms = 10
    pattern_interval_steps = int(pattern_interval_ms / dt)

    def get_spike_sequence(afferent_indices):
        spikes = []
        for i, afferent_idx in enumerate(afferent_indices):
            time_step = i * pattern_interval_steps
            spikes.append((time_step, afferent_idx))
        return spikes

    pattern_A_spikes = get_spike_sequence(pattern_A_afferents)
    pattern_B_spikes = get_spike_sequence(pattern_B_afferents)
    pattern_duration_steps = max(len(pattern_A_spikes), len(pattern_B_spikes)) * pattern_interval_steps
    
    return pattern_A_spikes, pattern_B_spikes, pattern_duration_steps, pattern_A_afferents, pattern_B_afferents

# --- train_layer1, train_layer2 수정: debug_sender -> buffer_manager 사용 ---
def train_layer1(layer, patterns, trials, dt, base_current, inhibition_current, buffer_manager):
    print("--- 1단계: Layer 1 경쟁 학습 시작 (Optimized) ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    pattern_choices = [pA_spikes, pB_spikes]

    base_current_gpu = cp.full(layer.num_neurons, base_current, dtype=DTYPE)
    last_axon_spikes = cp.zeros(layer.num_neurons, dtype=DTYPE)

    for trial in range(trials):
        pattern_spikes = pattern_choices[np.random.randint(len(pattern_choices))]
        layer.reset_state()
        
        for step in range(p_steps):
            presynaptic_spikes = cp.zeros(layer.num_afferents, dtype=DTYPE)
            for t, afferent_idx in pattern_spikes:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            
            result = layer.step_competitive(
                presynaptic_spikes,
                external_currents=base_current_gpu,
                inhibition_current=inhibition_current
            )
            last_axon_spikes = result.axon_spikes

        # 시각화를 위해 현재 상태 전송 (새로운 방식)
        buffer_manager.update_buffer("L1_MembranePotential", layer.membrane_potential)
        buffer_manager.update_buffer("L1_AxonSpikes", last_axon_spikes)

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")
    
    layer.input_learning_rate = 0.0
    print("--- 1단계: Layer 1 경쟁 학습 완료 (가중치 고정) ---")

def train_layer2(layer1, layer2, patterns, trials, dt, base_current, buffer_manager):
    print("\n--- 2단계: Layer 2 문맥 학습 시작 ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    
    context_sequence = [pA_spikes, pB_spikes]
    sequence_gap_steps = int(20 / dt)
    total_steps = p_steps * 2 + sequence_gap_steps

    for trial in range(trials):
        layer1.reset_state()
        layer2.reset_state()

        for step in range(total_steps):
            current_pattern_spikes = []
            if step < p_steps:
                current_pattern_spikes = pA_spikes
            elif step >= p_steps + sequence_gap_steps:
                current_pattern_spikes = pB_spikes
            
            l1_input_spikes = cp.zeros(layer1.num_afferents, dtype=DTYPE)
            step_in_pattern = step if step < p_steps else step - (p_steps + sequence_gap_steps)
            for t, afferent_idx in current_pattern_spikes:
                if step_in_pattern == t:
                    l1_input_spikes[afferent_idx] = 1.0

            base_current_gpu_l1 = cp.full(layer1.num_neurons, base_current, dtype=DTYPE)
            l1_result = layer1.step_competitive(
                l1_input_spikes,
                external_currents=base_current_gpu_l1,
                inhibition_current=-0.5
            )

            l2_currents = cp.full(layer2.num_neurons, 0.6, dtype=DTYPE)
            l2_result = layer2.step(l1_result.axon_spikes, external_currents=l2_currents)

            # 실시간 디버거로 데이터 전송 (새로운 방식)
            buffer_manager.update_buffer("L1_MembranePotential", layer1.membrane_potential)
            buffer_manager.update_buffer("L1_AxonSpikes", l1_result.axon_spikes)
            buffer_manager.update_buffer("L2_MembranePotential", layer2.membrane_potential)
            buffer_manager.update_buffer("L2_AxonSpikes", l2_result.axon_spikes)

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")
    
    layer2.input_learning_rate = 0.0
    print("--- 2단계: Layer 2 문맥 학습 완료 (가중치 고정) ---")

# (run_verification 함수는 변경 없음 - 그대로 사용)
def run_verification(layer1, layer2, patterns, dt, base_current):
    """학습된 네트워크의 문맥 이해도를 검증합니다."""
    print("\n--- 3단계: 학습 결과 검증 시작 ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    sequence_gap_steps = int(20 / dt)

    test_cases = {
        "'A -> B' 순서 (학습된 문맥)": [pA_spikes, pB_spikes],
        "'B -> A' 순서 (틀린 문맥)": [pB_spikes, pA_spikes],
        "'A' 패턴만 제시": [pA_spikes],
        "'B' 패턴만 제시": [pB_spikes]
    }

    for case_name, sequence in test_cases.items():
        layer1.reset_state()
        layer2.reset_state()
        
        total_steps = p_steps * len(sequence) + sequence_gap_steps * (len(sequence) - 1)
        if not sequence:
            total_steps = 0

        l2_total_spikes = 0

        for step in range(total_steps):
            current_pattern_spikes = []
            current_pattern_idx = step // (p_steps + sequence_gap_steps)
            if current_pattern_idx < len(sequence):
                current_pattern_spikes = sequence[current_pattern_idx]
                step_in_pattern = step % (p_steps + sequence_gap_steps)
            
                l1_input_spikes = cp.zeros(layer1.num_afferents, dtype=DTYPE)
                for t, afferent_idx in current_pattern_spikes:
                    if step_in_pattern == t:
                        l1_input_spikes[afferent_idx] = 1.0

                base_current_gpu_l1 = cp.full(layer1.num_neurons, base_current, dtype=DTYPE)
                l1_result = layer1.step_competitive(
                    l1_input_spikes,
                    external_currents=base_current_gpu_l1,
                    inhibition_current=-0.5
                )

                l2_currents = cp.full(layer2.num_neurons, 0.6, dtype=DTYPE)
                l2_result = layer2.step(l1_result.axon_spikes, external_currents=l2_currents)
                l2_total_spikes += cp.sum(l2_result.axon_spikes)

        print(f"  [질문] {case_name} -> [응답] Layer 2 총 발화 횟수: {int(l2_total_spikes)}")

def test_hierarchical_learning():
    if not GPU_AVAILABLE:
        print("계층적 학습 테스트: CUDA 장치가 없어 건너뜁니다.")
        return

    # --- Vulkan 및 버퍼 매니저 초기화 ---
    print("Initializing Vulkan for memory sharing...")
    # LveWindow와 LveDevice는 vkdbg 모듈이 내부적으로 사용하기 위한 Vulkan 컨텍스트를 생성합니다.
    # 이 객체들은 실제 윈도우를 띄우지 않을 수도 있습니다 (headless).
    # 이 부분은 실제 LveDevice 구현에 따라 달라집니다.
    try:
        lve_window = LveWindow(200, 200, "Vulkan Context")
        lve_device = LveDevice(lve_window)
        vkdbg.set_devices(lve_device.device_handle(), lve_device.physical_device_handle())
        print("Vulkan devices set in vkdbg module.")
    except Exception as e:
        print(f"Could not initialize Vulkan via Lve classes: {e}")
        print("This test requires a valid Vulkan context. Please check your setup.")
        # LveDevice/LveWindow를 pybind11로 래핑하거나, 핸들을 얻는 다른 방법이 필요합니다.
        # 지금은 이 부분이 실패하면 테스트를 진행할 수 없습니다.
        return

    buffer_manager = SharedBufferManager()

    np.random.seed(42)
    cp.random.seed(42)

    # --- 환경 설정 ---
    num_neurons_l1 = 20
    num_afferents_l1 = 10
    num_neurons_l2 = 5
    dt = 0.1
    base_current = 0.05
    inhibition_current = -0.5

    pA_spikes, pB_spikes, p_steps, pA_aff, pB_aff = define_patterns(num_afferents_l1, dt)
    patterns = {
        'pA_spikes': pA_spikes, 'pB_spikes': pB_spikes, 'p_steps': p_steps,
        'pA_aff': pA_aff, 'pB_aff': pB_aff
    }

    # --- Layer 1, 2 설정 (기존과 동일) ---
    l1_weights = cp.random.uniform(0.2, 0.5, size=(num_neurons_l1, num_afferents_l1)).astype(DTYPE)
    l1_weights[0:num_neurons_l1//2, pA_aff] += 0.3
    l1_weights[num_neurons_l1//2:, pB_aff] += 0.3
    layer1 = PyramidalLayer(
        num_neurons=num_neurons_l1, num_afferents=num_afferents_l1, num_branches=1,
        dt=dt, input_learning_rate=0.002, initial_input_weights=l1_weights
    )

    layer2 = PyramidalLayer(
        num_neurons=num_neurons_l2, num_afferents=num_neurons_l1, num_branches=1,
        dt=dt, input_learning_rate=0.01
    )
    l2_weights = cp.random.uniform(0.2, 0.5, size=(num_neurons_l2, num_neurons_l1)).astype(DTYPE)
    l1_A_specialists = cp.arange(0, num_neurons_l1 // 2)
    l2_weights[0, l1_A_specialists] += 0.3
    layer2.input_weights = l2_weights

    # --- 학습 및 검증 실행 (buffer_manager 전달) ---
    train_layer1(layer1, patterns, trials=200, dt=dt, base_current=base_current, inhibition_current=inhibition_current, buffer_manager=buffer_manager)
    verify_and_visualize_layer1(layer1, patterns, dt, base_current, inhibition_current)
    train_layer2(layer1, layer2, patterns, trials=300, dt=dt, base_current=base_current, buffer_manager=buffer_manager)
    run_verification(layer1, layer2, patterns, dt, base_current)

    # --- 자원 해제 ---
    # buffer_manager.cleanup() # finally 블록으로 이동
    return buffer_manager

if __name__ == "__main__":
    # buffer_manager를 try 블록보다 먼저 선언
    buffer_manager = None
    try:
        # test_hierarchical_learning 함수 내에서 buffer_manager가 생성됨
        buffer_manager = test_hierarchical_learning()
    finally:
        # 스크립트 종료 시 항상 자원 해제
        if buffer_manager:
            buffer_manager.cleanup()