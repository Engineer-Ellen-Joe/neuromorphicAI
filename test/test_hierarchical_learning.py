"""
SNN의 계층적 학습 및 문맥 이해 능력을 검증합니다.

이 스크립트는 3개의 뉴런으로 구성된 2계층 네트워크를 시뮬레이션합니다.
학습은 두 단계로 분리하여 안정성을 확보합니다.

- 1단계 (1계층 학습): '경쟁 학습'을 통해 뉴런 A와 B가 각각 '패턴 A'와
  '패턴 B'의 전문가로 자율적으로 분화합니다.

- 2단계 (2계층 학습): 학습된 1계층의 출력을 입력으로 받아, 뉴런 C가
  'A 발화 -> B 발화'라는 특정 순서, 즉 '문맥'을 STDP 규칙으로 학습합니다.

학습 완료 후, 네트워크에 여러 종류의 질문(패턴 순서)을 던져
각 뉴런의 반응을 확인함으로써, 네트워크가 문맥을 올바르게 학습했는지 검증합니다.
"""

import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_neuron import PyramidalNeuron

# --- 기본 설정 ---
GPU_AVAILABLE = cp.cuda.is_available()
try:
    plt.rcParams['font.family'] = 'Noto Sans KR'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("Noto Sans KR 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    pass

def _to_numpy(array):
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)

def create_base_neuron(num_afferents, dt, random_seed, input_lr=0.015, output_lr=0.0):
    """실험에 사용할 기본 뉴런 모델을 생성합니다."""
    return PyramidalNeuron(
        num_afferents=num_afferents,
        num_branches=1,
        dt=dt,
        membrane_time_constant=12e-3,
        membrane_capacitance=180e-12,
        leak_potential=-70e-3,
        reset_potential=-68e-3,
        ais_threshold=-61e-3,
        refractory_period=1e-3,
        input_learning_rate=input_lr,
        initial_input_weights=np.random.normal(loc=0.4, scale=0.05, size=num_afferents),
        stdp_tau_pre=12e-3,
        stdp_tau_post=12e-3,
        output_learning_rate=output_lr,
        random_state=random_seed,
    )

def define_patterns(num_afferents, dt):
    """학습에 사용할 두 개의 시공간 패턴을 정의합니다."""
    pattern_A_afferents = [3, 8, 14]
    pattern_B_afferents = [5, 10, 16]
    
    pattern_interval_ms = 10
    pattern_interval_steps = int(pattern_interval_ms * 1e-3 / dt)

    def get_spike_sequence(afferent_indices):
        spikes = []
        for i, afferent_idx in enumerate(afferent_indices):
            time_step = i * pattern_interval_steps
            spikes.append((time_step, afferent_idx))
        return spikes

    pattern_A_spikes = get_spike_sequence(pattern_A_afferents)
    pattern_B_spikes = get_spike_sequence(pattern_B_afferents)
    pattern_duration_steps = len(pattern_A_afferents) * pattern_interval_steps
    
    return pattern_A_spikes, pattern_B_spikes, pattern_duration_steps, pattern_A_afferents, pattern_B_afferents

def run_learning_L1(neurons, patterns, trials, dt):
    """1계층 뉴런들의 역할 분담을 위한 경쟁 학습을 수행합니다."""
    print("--- 1단계: 1계층 경쟁 학습 시작 ---")
    num_neurons = len(neurons)
    num_afferents = neurons[0].num_afferents
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    pattern_choices = [pA_spikes, pB_spikes]

    base_current = 80e-12
    inhibition_current = -200e-12
    inhibition_duration_steps = int(2e-3 / dt)

    for trial in range(trials):
        pattern_spikes = pattern_choices[np.random.randint(len(pattern_choices))]
        
        for n in neurons: n.reset_state()
        inhibition_timers = np.zeros(num_neurons)

        for step in range(p_steps):
            presynaptic_spikes = np.zeros(num_afferents)
            for t, afferent_idx in pattern_spikes:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            
            inhibition_timers -= 1
            winner_found = False

            for i, neuron in enumerate(neurons):
                if winner_found: continue
                current = inhibition_current if inhibition_timers[i] > 0 else base_current
                result = neuron.step(presynaptic_spikes, external_current=current)
                if result.axon_spike > 0:
                    winner_found = True
                    for j in range(num_neurons):
                        if i != j:
                            inhibition_timers[j] = inhibition_duration_steps
        
        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")
    print("--- 1단계: 1계층 경쟁 학습 완료 ---")

def run_learning_L2(neurons_L1, neuron_L2, patterns, trials, dt):
    """2계층 뉴런의 문맥 학습을 수행합니다."""
    print("\n--- 2단계: 2계층 문맥 학습 시작 ---")
    num_neurons_L1 = len(neurons_L1)
    num_afferents_L1 = neurons_L1[0].num_afferents
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    
    base_current = 80e-12
    context_sequence = [pA_spikes, pB_spikes]
    sequence_gap_steps = int(20 * 1e-3 / dt)
    total_steps = p_steps * 2 + sequence_gap_steps

    # 1계층은 더 이상 학습하지 않음
    for n in neurons_L1: n.input_learning_rate_gpu = cp.float64(0.0)

    for trial in range(trials):
        for n in neurons_L1: n.reset_state()
        neuron_L2.reset_state()
        
        L1_outputs = np.zeros((total_steps, num_neurons_L1))

        # 1계층은 학습 없이 '반응'만 함
        for i, pattern_spikes in enumerate(context_sequence):
            start_step = i * (p_steps + sequence_gap_steps)
            end_step = start_step + p_steps
            for step in range(start_step, end_step):
                presynaptic_spikes = np.zeros(num_afferents_L1)
                for t, afferent_idx in pattern_spikes:
                    if (step - start_step) == t:
                        presynaptic_spikes[afferent_idx] = 1.0
                
                for n_idx, neuron in enumerate(neurons_L1):
                    result = neuron.step(presynaptic_spikes, external_current=base_current)
                    if result.axon_spike > 0:
                        L1_outputs[step, n_idx] = 1.0
        
        # 2계층은 1계층의 반응을 보고 '학습'함
        for step in range(total_steps):
            neuron_L2.step(L1_outputs[step, :], external_current=base_current)

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")
    print("--- 2단계: 2계층 문맥 학습 완료 ---")


def run_verification(neurons, neuron_C, patterns, dt):
    """학습된 네트워크의 문맥 이해도를 검증합니다."""
    print("\n--- 3단계: 학습 결과 검증 시작 ---")
    
    pA_spikes, pB_spikes, p_steps = patterns['pA_spikes'], patterns['pB_spikes'], patterns['p_steps']
    num_afferents = neurons[0].num_afferents
    num_neurons_L1 = len(neurons)
    base_current = 80e-12
    sequence_gap_steps = int(20 * 1e-3 / dt)

    test_cases = {
        "'A' 패턴만 제시": [pA_spikes],
        "'B' 패턴만 제시": [pB_spikes],
        "'A -> B' 순서로 제시 (학습된 문맥)": [pA_spikes, pB_spikes],
        "'B -> A' 순서로 제시 (틀린 문맥)": [pB_spikes, pA_spikes]
    }

    # 검증 시에는 모든 뉴런의 학습 기능 비활성화
    for n in neurons: n.input_learning_rate_gpu = cp.float64(0.0)
    neuron_C.input_learning_rate_gpu = cp.float64(0.0)

    # 뉴런 A가 패턴 A 전문가, 뉴런 B가 패턴 B 전문가라고 가정
    # 초기 시드값에 따라 역할이 바뀔 수 있으므로, 학습 후 확인 필요
    neuron_A_expert_for_A = _to_numpy(neurons[0].input_weights)[patterns['pA_aff']].mean() > _to_numpy(neurons[1].input_weights)[patterns['pA_aff']].mean()
    neuron_A_idx = 0 if neuron_A_expert_for_A else 1
    neuron_B_idx = 1 if neuron_A_expert_for_A else 0

    print(f"(정보) 뉴런 {neuron_A_idx+1}이(가) 패턴 A 전문가로, 뉴런 {neuron_B_idx+1}이(가) 패턴 B 전문가로 결정되었습니다.")


    for case_name, sequence in test_cases.items():
        print(f"\n[질문] {case_name}")
        
        for n in neurons: n.reset_state()
        neuron_C.reset_state()
        
        total_steps = p_steps * len(sequence) + sequence_gap_steps * (len(sequence) - 1)
        if total_steps == 0: total_steps = p_steps

        L1_outputs = np.zeros((total_steps, num_neurons_L1))
        L1_spike_counts = np.zeros(num_neurons_L1)
        L2_spike_count = 0

        for i, pattern_spikes in enumerate(sequence):
            start_step = i * (p_steps + sequence_gap_steps)
            end_step = start_step + p_steps
            for step in range(start_step, end_step):
                presynaptic_spikes = np.zeros(num_afferents)
                for t, afferent_idx in pattern_spikes:
                    if (step - start_step) == t:
                        presynaptic_spikes[afferent_idx] = 1.0
                
                for n_idx, neuron in enumerate(neurons):
                    result = neuron.step(presynaptic_spikes, external_current=base_current)
                    if result.axon_spike > 0:
                        L1_outputs[step, n_idx] = 1.0
                        L1_spike_counts[n_idx] += 1
        
        for step in range(total_steps):
            result = neuron_C.step(L1_outputs[step, :], external_current=base_current)
            if result.axon_spike > 0:
                L2_spike_count += 1

        print("  [응답]")
        print(f"    - 'A' 전문가 뉴런({neuron_A_idx+1}) 발화 횟수: {int(L1_spike_counts[neuron_A_idx])}")
        print(f"    - 'B' 전문가 뉴런({neuron_B_idx+1}) 발화 횟수: {int(L1_spike_counts[neuron_B_idx])}")
        print(f"    - '문맥' 전문가 뉴런(C) 발화 횟수: {int(L2_spike_count)}")


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("계층적 학습 테스트: CUDA 장치가 없어 건너뜁니다.")
    else:
        # 1. 환경 설정
        num_afferents_L1 = 20
        num_afferents_L2 = 2
        dt = 1e-4
        
        neurons_L1 = [create_base_neuron(num_afferents_L1, dt, 42), 
                      create_base_neuron(num_afferents_L1, dt, 1337)]
        neuron_L2 = create_base_neuron(num_afferents_L2, dt, 777, input_lr=0.02)
        
        pA_spikes, pB_spikes, p_steps, pA_aff, pB_aff = define_patterns(num_afferents_L1, dt)
        pattern_data = {
            'pA_spikes': pA_spikes, 'pB_spikes': pB_spikes, 'p_steps': p_steps,
            'pA_aff': pA_aff, 'pB_aff': pB_aff
        }

        # 2. 학습 실행 (단계별 분리)
        run_learning_L1(neurons_L1, pattern_data, trials=200, dt=dt)
        run_learning_L2(neurons_L1, neuron_L2, pattern_data, trials=200, dt=dt)

        # 3. 검증 실행
        run_verification(neurons_L1, neuron_L2, pattern_data, dt)