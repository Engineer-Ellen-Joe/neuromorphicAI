"""
2-Layer SNN의 계층적 학습(Hierarchical Learning)을 검증합니다.

이 스크립트는 2개의 PyramidalLayer를 사용하여, 하위 계층에서 학습된 패턴 인식을
기반으로 상위 계층이 더 추상적인 '문맥(context)' 또는 '순서(sequence)'를
학습하는 과정을 시뮬레이션합니다.

- 1단계 (Layer 1 학습): 경쟁 학습을 통해 Layer 1의 뉴런들이 각각 특정
  입력 패턴(A, B)의 전문가로 분화합니다. (test_network_competition.py의 검증된 로직 사용)

- 2단계 (Layer 2 학습): 학습이 완료된 Layer 1의 출력을 입력으로 받아,
  Layer 2가 'A패턴 -> B패턴'이라는 특정 순서를 STDP 규칙으로 학습합니다.

- 3단계 (검증): 학습된 전체 네트워크에 여러 종류의 질문(패턴 순서)을 던져
  Layer 2가 오직 학습된 순서에만 반응하는지 확인하여, 네트워크가 '문맥'을
  올바르게 학습했는지 검증합니다.
"""

import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_layer import PyramidalLayer, DTYPE
from src.snn_visualizer_client import SNNVisualizer

# --- 기본 설정 ---
GPU_AVAILABLE = cp.cuda.is_available()

def verify_and_visualize_layer1(layer, patterns, dt, base_current, inhibition_current):
    """학습된 1계층의 역할 분담 결과를 검증하고 시각화합니다."""
    print("\n--- 중간 검증: Layer 1 역할 분담 확인 ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    
    spike_counts = cp.zeros((2, layer.num_neurons), dtype=cp.int32) # 0: Pattern A, 1: Pattern B
    test_patterns = [pA_spikes, pB_spikes]
    pattern_names = ["Pattern A", "Pattern B"]
    
    base_current_gpu = cp.full(layer.num_neurons, 0.2, dtype=DTYPE) # 검증을 위해 탐침 전류를 미세 조정

    # 학습 비활성화 상태에서 각 패턴에 대한 반응 테스트
    original_lr = layer.input_learning_rate
    layer.input_learning_rate = 0.0

    for i, pattern_spikes in enumerate(test_patterns):
        layer.reset_state()
        for _ in range(5): # 5번 반복하여 평균적인 반응 확인
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

    # 학습률 원상복구
    layer.input_learning_rate = original_lr

    # --- 시각화 ---
    spike_counts_cpu = spike_counts.get()
    neuron_indices = np.arange(layer.num_neurons)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.4
    
    # A-specialists should be the first half, B-specialists the second half
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

def train_layer1(layer, patterns, trials, dt, base_current, inhibition_current, visualizer=None):
    """1계층 뉴런들의 역할 분담을 위한 경쟁 학습을 수행합니다. (Optimized)"""
    print("--- 1단계: Layer 1 경쟁 학습 시작 (Optimized) ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    pattern_choices = [pA_spikes, pB_spikes]

    # Prepare a constant external current array on the GPU
    base_current_gpu = cp.full(layer.num_neurons, base_current, dtype=DTYPE)

    num_afferents_l1 = layer.num_afferents
    num_neurons_l1 = layer.num_neurons
    # For visualization, we assume L2 is not active yet.
    num_neurons_l2 = visualizer.layer_structure[2] if visualizer else 0

    for trial in range(trials):
        pattern_spikes = pattern_choices[np.random.randint(len(pattern_choices))]
        layer.reset_state()
        
        for step in range(p_steps):
            # Create presynaptic spikes for the current time step
            presynaptic_spikes = cp.zeros(layer.num_afferents, dtype=DTYPE)
            for t, afferent_idx in pattern_spikes:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            
            # Call the new, optimized competitive step method
            result = layer.step_competitive(
                presynaptic_spikes,
                external_currents=base_current_gpu,
                inhibition_current=inhibition_current
            )

        if visualizer:
            # Combine neuron states for visualization
            # Input layer spikes + Layer 1 spikes + Layer 2 (zeros)
            vis_neuron_states = np.concatenate([
                presynaptic_spikes.get(),
                (result.axon_spikes > 0).get().astype(np.float32),
                np.zeros(num_neurons_l2, dtype=np.float32)
            ])
            # Combine weights for visualization
            l1_w = layer.input_weights.get().flatten()
            # L2 weights are not trained yet, so we can send zeros or initial

            l2_w_size = num_neurons_l1 * num_neurons_l2
            l2_w = np.zeros(l2_w_size, dtype=np.float32)

            vis_weights = np.concatenate([l1_w, l2_w])

            visualizer.update(
                neuron_states=vis_neuron_states,
                weights=vis_weights
            )

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")
    
    layer.input_learning_rate = 0.0 # Layer 1 학습 고정
    print("--- 1단계: Layer 1 경쟁 학습 완료 (가중치 고정) ---")

def train_layer2(layer1, layer2, patterns, trials, dt, base_current, visualizer=None):
    """2계층 뉴런의 문맥 학습을 수행합니다."""
    print("\n--- 2단계: Layer 2 문맥 학습 시작 ---")
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    
    context_sequence = [pA_spikes, pB_spikes]
    sequence_gap_steps = int(20 / dt) # 20ms 갭
    total_steps = p_steps * 2 + sequence_gap_steps

    for trial in range(trials):
        layer1.reset_state()
        layer2.reset_state()
        
        # Layer 1은 경쟁만 하고 학습은 안함 (가중치 고정됨)
        inhibition_l1 = cp.zeros(layer1.num_neurons, dtype=DTYPE)

        for step in range(total_steps):
            # 현재 스텝에 맞는 패턴 입력 생성
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

            # Layer 1 실행 (Optimized)
            base_current_gpu_l1 = cp.full(layer1.num_neurons, base_current, dtype=DTYPE)
            l1_result = layer1.step_competitive(
                l1_input_spikes,
                external_currents=base_current_gpu_l1,
                inhibition_current=-0.5  # L1의 경쟁을 위한 억제 전류
            )

            # Layer 2 실행 (Layer 1의 출력을 입력으로 받음)
            l2_currents = cp.full(layer2.num_neurons, 0.6, dtype=DTYPE) # L2 자극 미세 조정
            # layer2.step(l1_result.axon_spikes, external_currents=l2_currents)
            l2_result = layer2.step(l1_result.axon_spikes, external_currents=l2_currents)

            if visualizer:
                vis_neuron_states = np.concatenate([
                    l1_input_spikes.get(),
                    (l1_result.axon_spikes > 0).get().astype(np.float32),
                    (l2_result.axon_spikes > 0).get().astype(np.float32)
                ])

                l1_w = layer1.input_weights.get().flatten()
                l2_w = layer2.input_weights.get().flatten()
                vis_weights = np.concatenate([l1_w, l2_w])

                visualizer.update(
                    neuron_states=vis_neuron_states,
                    weights=vis_weights
                )

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")
    
    layer2.input_learning_rate = 0.0 # Layer 2 학습 고정
    print("--- 2단계: Layer 2 문맥 학습 완료 (가중치 고정) ---")

def run_verification(layer1, layer2, patterns, dt, base_current, visualizer=None):
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

    # Get final weights for visualization during verification
    if visualizer:
        l1_w = layer1.input_weights.get().flatten()
        l2_w = layer2.input_weights.get().flatten()
        vis_weights = np.concatenate([l1_w, l2_w])

    for case_name, sequence in test_cases.items():
        layer1.reset_state()
        layer2.reset_state()
        
        total_steps = p_steps * len(sequence) + sequence_gap_steps * (len(sequence) - 1)
        if not sequence:
            total_steps = 0

        l2_total_spikes = 0
        inhibition_l1 = cp.zeros(layer1.num_neurons, dtype=DTYPE)

        print(f"\n  [질문] {case_name}")

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

                # Layer 1 실행 (Optimized, 경쟁만)
                base_current_gpu_l1 = cp.full(layer1.num_neurons, base_current, dtype=DTYPE)
                l1_result = layer1.step_competitive(
                    l1_input_spikes,
                    external_currents=base_current_gpu_l1,
                    inhibition_current=-0.5 # L1의 경쟁을 위한 억제 전류
                )

                # Layer 2 실행 (학습 없이 반응만)
                l2_currents = cp.full(layer2.num_neurons, 0.6, dtype=DTYPE) # L2 자극 미세 조정
                l2_result = layer2.step(l1_result.axon_spikes, external_currents=l2_currents)
                l2_total_spikes += cp.sum(l2_result.axon_spikes)

                if visualizer:
                    vis_neuron_states = np.concatenate([
                        l1_input_spikes.get(),
                        (l1_result.axon_spikes > 0).get().astype(np.float32),
                        (l2_result.axon_spikes > 0).get().astype(np.float32)
                    ])
                    # Weights are fixed during verification, so send the final weights
                    visualizer.update(
                        neuron_states=vis_neuron_states,
                        weights=vis_weights
                    )

        print(f"  [질문] {case_name} -> [응답] Layer 2 총 발화 횟수: {int(l2_total_spikes)}")

def test_hierarchical_learning():
    if not GPU_AVAILABLE:
        print("계층적 학습 테스트: CUDA 장치가 없어 건너뜁니다.")
        return

    np.random.seed(42)
    cp.random.seed(42)

    # --- 환경 설정 ---
    num_neurons_l1 = 20
    num_afferents_l1 = 10
    num_neurons_l2 = 5
    dt = 0.1  # ms
    base_current = 0.05 # nA
    inhibition_current = -0.5 # nA

    pA_spikes, pB_spikes, p_steps, pA_aff, pB_aff = define_patterns(num_afferents_l1, dt)
    patterns = {
        'pA_spikes': pA_spikes, 'pB_spikes': pB_spikes, 'p_steps': p_steps,
        'pA_aff': pA_aff, 'pB_aff': pB_aff
    }

    # --- Layer 1 설정 (경쟁 학습용) ---
    l1_weights_init = cp.random.uniform(0.2, 0.5, size=(num_neurons_l1, num_afferents_l1)).astype(DTYPE)
    l1_weights_init[0:num_neurons_l1//2, pA_aff] += 0.3
    l1_weights_init[num_neurons_l1//2:, pB_aff] += 0.3
    layer1 = PyramidalLayer(
        num_neurons=num_neurons_l1, num_afferents=num_afferents_l1, num_branches=1,
        dt=dt, input_learning_rate=0.002, initial_input_weights=l1_weights_init
    )

    # --- Layer 2 설정 (문맥 학습용) ---
    layer2 = PyramidalLayer(
        num_neurons=num_neurons_l2, num_afferents=num_neurons_l1, num_branches=1,
        dt=dt, input_learning_rate=0.01
    )
    # --- Layer 2 초기 편향 추가 ---
    # L2의 0번 뉴런이 L1의 A-전문가 뉴런에 더 민감하게 반응하도록 설정
    l2_weights_init = cp.random.uniform(0.2, 0.5, size=(num_neurons_l2, num_neurons_l1)).astype(DTYPE)
    l1_A_specialists = cp.arange(0, num_neurons_l1 // 2)
    l2_weights_init[0, l1_A_specialists] += 0.3
    layer2.input_weights = l2_weights_init

    # --- SNN Visualizer 초기화 ---
    visualizer = None
    try:
        print("\n--- Visualizer 초기화 시작 ---")
        # 시각화를 위한 네트워크 구조 정의
        vis_layer_structure = [num_afferents_l1, num_neurons_l1, num_neurons_l2]

        # 연결 정보 추출 (입력 -> L1)
        l1_sources, l1_targets = np.mgrid[0:num_afferents_l1, 0:num_neurons_l1]
        l1_s = l1_sources.flatten()
        l1_t = l1_targets.flatten() + num_afferents_l1 # L1 뉴런 인덱스 오프셋
        l1_w = layer1.input_weights.get().flatten()

        # 연결 정보 추출 (L1 -> L2)
        l2_sources, l2_targets = np.mgrid[0:num_neurons_l1, 0:num_neurons_l2]
        l2_s = l2_sources.flatten() + num_afferents_l1 # L1 뉴런 인덱스 오프셋
        l2_t = l2_targets.flatten() + num_afferents_l1 + num_neurons_l1 # L2 뉴런 인덱스 오프셋
        l2_w = layer2.input_weights.get().flatten()

        # 시각화를 위한 전체 연결 정보 취합
        vis_connections = (
            np.concatenate([l1_s, l2_s]).astype(np.int32).tolist(),
            np.concatenate([l1_t, l2_t]).astype(np.int32).tolist(),
            np.concatenate([l1_w, l2_w]).astype(np.float32).tolist()
        )

        # 입력 시냅스는 없으므로 빈 리스트 전달
        vis_input_synapses = ([], [])

        visualizer = SNNVisualizer(
            layer_structure=vis_layer_structure,
            connections=vis_connections,
            input_synapses=vis_input_synapses
        )
        print("--- Visualizer 초기화 완료 ---")

        # --- 학습 및 검증 실행 ---
        train_layer1(layer1, patterns, trials=200, dt=dt, base_current=base_current, inhibition_current=inhibition_current, visualizer=visualizer)
        verify_and_visualize_layer1(layer1, patterns, dt, base_current, inhibition_current)
        train_layer2(layer1, layer2, patterns, trials=300, dt=dt, base_current=base_current, visualizer=visualizer)
        run_verification(layer1, layer2, patterns, dt, base_current)
    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
    finally:
        if visualizer:
            visualizer.close()
            print("--- Visualizer 종료 ---")

if __name__ == "__main__":
    test_hierarchical_learning()
