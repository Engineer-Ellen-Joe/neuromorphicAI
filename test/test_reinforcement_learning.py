"""
PyramidalNeuron의 강화학습 능력을 검증합니다.

이 스크립트는 두 단계의 학습 과정을 통해 뉴런이 보상을 최대화하는 행동을
학습하는 과정을 시뮬레이션합니다.

1. STDP 기반 사전학습: 뉴런이 두 개의 다른 시공간 패턴('A', 'B')을
   모두 '인식'할 수 있도록 입력 시냅스 가중치를 학습합니다.

2. BCM 기반 강화학습: 'A' 패턴에만 발화했을 때 '보상'을 주어, 뉴런이
   해당 행동을 선택하도록 출력 시냅스 가중치를 학습합니다. '보상'은
   BCM 학습률을 일시적으로 높여주는 '도파민'과 같은 역할을 합니다.

실행 전, `matplotlib` 라이브러리가 설치되어 있어야 합니다:
pip install matplotlib
"""

import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

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


def create_base_neuron(num_afferents, dt):
    """실험에 사용할 기본 뉴런 모델을 생성합니다."""
    return PyramidalNeuron(
        num_afferents=num_afferents,
        num_branches=1, # 단일 행동 출력을 위해 branch는 1개
        dt=dt,
        membrane_time_constant=12e-3,
        membrane_capacitance=180e-12,
        leak_potential=-70e-3,
        reset_potential=-68e-3,
        ais_threshold=-61e-3,
        refractory_period=1e-3,
        # STDP 사전학습을 위한 파라미터
        input_learning_rate=0.015,
        initial_input_weights=np.full(num_afferents, 0.4),
        stdp_tau_pre=12e-3,
        stdp_tau_post=12e-3,
        # 강화학습을 위한 파라미터 (초기에는 비활성화)
        output_learning_rate=0.0,
        initial_output_weights=[0.5],
        bcm_tau=1.0, # BCM 파라미터
        branch_activity_tau=50e-3,
        random_state=42,
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

def run_pre_training(neuron, patterns_to_train, presentation_steps):
    """STDP를 사용하여 여러 패턴의 입력 가중치를 각각 순차적으로 사전 학습시킵니다."""
    print("--- 1단계: STDP 사전학습 시작 ---")
    epochs_per_pattern = 120  # 각 패턴당 학습 횟수
    training_current = 350e-12
    
    # 각 패턴에 대해 순차적으로 학습 진행
    for i, pattern_spikes in enumerate(patterns_to_train):
        print(f"  패턴 {i+1} 학습 중...")
        for epoch in range(epochs_per_pattern):
            neuron.reset_state() # 뉴런의 전위 등 상태만 리셋
            for step in range(presentation_steps):
                presynaptic_spikes = np.zeros(neuron.num_afferents)
                for t, afferent_idx in pattern_spikes:
                    if step == t:
                        presynaptic_spikes[afferent_idx] = 1.0
                neuron.step(presynaptic_spikes, external_current=training_current)
            if (epoch + 1) % 40 == 0:
                print(f"    - Epoch {epoch+1}/{epochs_per_pattern}")

    print("--- 1단계: STDP 사전학습 완료 ---")
    return _to_numpy(neuron.input_weights).copy()

def run_reinforcement_learning(neuron, patterns_with_labels, presentation_steps):
    """BCM과 보상 신호를 사용하여 출력 가중치를 학습시킵니다."""
    print("--- 2단계: 강화학습 시작 ---")
    # 입력 가중치 동결
    neuron.input_learning_rate_gpu = cp.float64(0.0)
    
    trials = 500
    reward_lr = 0.1  # 보상(도파민) 강도
    validation_current = 80e-12
    
    output_weight_history = [_to_numpy(neuron.output_weights)[0]]
    correct_actions = []

    for trial in range(trials):
        neuron.reset_state()
        is_pattern_A, pattern_spikes = patterns_with_labels[np.random.randint(len(patterns_with_labels))]
        
        spike_fired = False
        for step in range(presentation_steps):
            presynaptic_spikes = np.zeros(neuron.num_afferents)
            for t, afferent_idx in pattern_spikes:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            result = neuron.step(presynaptic_spikes, external_current=validation_current)
            if result.axon_spike > 0:
                spike_fired = True

        # 보상 규칙 적용
        # 보상 규칙 적용
        action_is_correct = (is_pattern_A and spike_fired) or (not is_pattern_A and not spike_fired)
        correct_actions.append(1 if action_is_correct else 0)

        # 직접 가중치 수정 (미세 조정)
        if action_is_correct:
            # 성공 시 출력 가중치(정책)를 점진적으로 강화
            neuron.output_weights[0] += 0.005
        else:
            # 실패 시 출력 가중치를 점진적으로 약화
            neuron.output_weights[0] -= 0.002
        
        # 가중치가 경계를 벗어나지 않도록 제한
        if neuron.output_weights[0] > 1.5:
            neuron.output_weights[0] = cp.float64(1.5)
        elif neuron.output_weights[0] < 0.0:
            neuron.output_weights[0] = cp.float64(0.0)

        output_weight_history.append(_to_numpy(neuron.output_weights)[0])
        if (trial + 1) % 50 == 0:
            print(f"  강화학습 Trial {trial+1}/{trials} 완료... (최근 50회 성공률: {np.mean(correct_actions[-50:])*100:.1f}%)")

    print("--- 2단계: 강화학습 완료 ---")
    return output_weight_history, correct_actions


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("강화학습 테스트: CUDA 장치가 없어 건너뜁니다.")
    else:
        # 1. 환경 설정
        num_afferents = 20
        dt = 1e-4
        neuron = create_base_neuron(num_afferents, dt)
        pA_spikes, pB_spikes, p_steps, pA_aff, pB_aff = define_patterns(num_afferents, dt)

        # 2. 1단계: STDP 사전학습 실행
        pre_trained_weights = run_pre_training(neuron, [pA_spikes, pB_spikes], p_steps)
        
        # 3. 2단계: 강화학습 실행
        patterns_with_labels = [(True, pA_spikes), (False, pB_spikes)]
        output_weight_history, correct_actions = run_reinforcement_learning(neuron, patterns_with_labels, p_steps)

        # 4. 결과 시각화
        print("결과 시각화 중...")
        fig = plt.figure(figsize=(14, 18))
        gs = fig.add_gridspec(3, 1)
        fig.suptitle("강화학습(Dopamine-like Reward) 결과", fontsize=18)

        # 사전학습 결과 그래프
        ax1 = fig.add_subplot(gs[0])
        bar_colors = ['red' if i in pA_aff else 'blue' if i in pB_aff else 'grey' for i in range(num_afferents)]
        ax1.bar(range(num_afferents), pre_trained_weights, color=bar_colors)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Pattern A Afferents'),
                           Patch(facecolor='blue', label='Pattern B Afferents')]
        ax1.legend(handles=legend_elements)

        ax1.set_title("1단계: STDP 사전학습 후 입력 가중치", fontsize=14)
        ax1.set_xlabel("시냅스 인덱스")
        ax1.set_ylabel("가중치")

        # 강화학습 결과 그래프 (출력 가중치)
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(output_weight_history, label="출력 가중치 (Policy)")
        ax2.set_title("2단계: 강화학습에 따른 출력 가중치 변화", fontsize=14)
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("가중치")
        ax2.grid(True)
        ax2.legend()

        # 성공률 그래프
        ax3 = fig.add_subplot(gs[2])
        # 이동 평균 계산
        window_size = 50
        moving_avg = np.convolve(correct_actions, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(moving_avg, label="성공률 (이동 평균)")
        ax3.set_title("시간에 따른 행동 성공률 변화", fontsize=14)
        ax3.set_xlabel(f"Trial (x{window_size})")
        ax3.set_ylabel("성공률")
        ax3.set_ylim(0, 1.1)
        ax3.axhline(0.5, color='gray', linestyle='--', label="무작위 선택 (50%)")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("reinforcement_learning_result.png")
        plt.show()
        print("완료. reinforcement_learning_result.png 파일이 저장되었습니다.")
