"""
두 개의 PyramidalNeuron이 경쟁하며 역할을 분담하는 과정을 검증합니다.

이 스크립트는 '경쟁 학습(Competitive Learning)' 시나리오를 시뮬레이션합니다.
두 개의 뉴런이 동일한 입력을 공유하며, 한 뉴런이 발화하면 다른 뉴런을
억제하는 '측면 억제(Lateral Inhibition)' 규칙을 적용합니다.

이 경쟁을 통해, 각 뉴런은 서로 다른 입력 패턴에 대한 '전문가'로 분화하게 됩니다.
학습은 외부의 보상 신호 없이 STDP 규칙에 의해 자율적으로 이루어집니다.

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

def create_base_neuron(num_afferents, dt, random_seed):
    """실험에 사용할 기본 뉴런 모델을 생성합니다."""
    # 각 뉴런이 약간 다른 초기 조건을 갖도록 random_seed를 사용
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
        input_learning_rate=0.015,
        # 초기 가중치에 약간의 노이즈를 주어 경쟁의 시작점을 만듦
        initial_input_weights=np.random.normal(loc=0.4, scale=0.05, size=num_afferents),
        stdp_tau_pre=12e-3,
        stdp_tau_post=12e-3,
        output_learning_rate=0.0,
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


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("경쟁 학습 테스트: CUDA 장치가 없어 건너뜁니다.")
    else:
        # 1. 환경 설정
        num_afferents = 20
        dt = 1e-4
        num_neurons = 2
        
        neurons = [create_base_neuron(num_afferents, dt, seed) for seed in [42, 1337]]
        pA_spikes, pB_spikes, p_steps, pA_aff, pB_aff = define_patterns(num_afferents, dt)
        patterns = [pA_spikes, pB_spikes]

        # 2. 경쟁 학습 시뮬레이션
        print("--- 경쟁 학습 시뮬레이션 시작 ---")
        trials = 400
        base_current = 80e-12
        inhibition_current = -200e-12
        inhibition_duration_steps = int(2e-3 / dt)

        # 가중치 변화 기록
        weight_history = [[_to_numpy(n.input_weights).copy()] for n in neurons]

        for trial in range(trials):
            # 무작위로 패턴 선택
            pattern_spikes = patterns[np.random.randint(len(patterns))]
            
            # 뉴런 상태 리셋
            for n in neurons:
                n.reset_state()
            
            inhibition_timers = np.zeros(num_neurons)

            for step in range(p_steps):
                presynaptic_spikes = np.zeros(num_afferents)
                for t, afferent_idx in pattern_spikes:
                    if step == t:
                        presynaptic_spikes[afferent_idx] = 1.0
                
                inhibition_timers -= 1
                winner_found = False

                for i, neuron in enumerate(neurons):
                    if winner_found: # 한 스텝에 한 뉴런만 발화 가능
                        continue

                    current = inhibition_current if inhibition_timers[i] > 0 else base_current
                    result = neuron.step(presynaptic_spikes, external_current=current)

                    if result.axon_spike > 0:
                        winner_found = True
                        # 다른 모든 뉴런에게 억제 신호 전달
                        for j in range(num_neurons):
                            if i != j:
                                inhibition_timers[j] = inhibition_duration_steps
            
            # 가중치 기록
            for i in range(num_neurons):
                weight_history[i].append(_to_numpy(neurons[i].input_weights).copy())

            if (trial + 1) % 40 == 0:
                print(f"  Trial {trial+1}/{trials} 완료...")

        print("--- 경쟁 학습 완료 ---")

        # 3. 결과 시각화
        print("결과 시각화 중...")
        fig, axes = plt.subplots(num_neurons, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("경쟁 학습 후 뉴런별 전문화 결과", fontsize=16)

        final_weights = [history[-1] for history in weight_history]

        for i, ax in enumerate(axes):
            bar_colors = ['red' if j in pA_aff else 'blue' if j in pB_aff else 'grey' for j in range(num_afferents)]
            ax.bar(range(num_afferents), final_weights[i], color=bar_colors)
            ax.set_title(f"뉴런 {i+1}의 최종 입력 가중치")
            ax.set_ylabel("가중치")
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Pattern A Afferents'),
                           Patch(facecolor='blue', label='Pattern B Afferents')]
        axes[0].legend(handles=legend_elements)
        axes[-1].set_xlabel("시냅스 인덱스")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("network_competition_result.png")
        plt.show()
        print("완료. network_competition_result.png 파일이 저장되었습니다.")
