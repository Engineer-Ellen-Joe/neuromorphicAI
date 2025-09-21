import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

# 프로젝트 루트를 경로에 추가하여 src 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.neuro_pyramidal_nrml import (
    CuPyNetwork,
    SomaParams,
    AISParams,
    CouplingParams,
    SynapseParams
)

def run_pattern_recognition_test():
    """
    STDP를 이용한 스파이크 시간 패턴 인식 학습 테스트.
    - 특정 뉴런 그룹이 발화할 때 출력 뉴런이 함께 발화하도록 학습 (상관관계 기반 학습).
    """
    # 1. 시뮬레이션 파라미터 (단위: ms, mV, pA, nS, pF)
    T_SIM = 5000.0  # 총 시뮬레이션 시간 (ms) - 5초
    DT = 0.1         # 시간 해상도 (ms)

    print("===== STDP 패턴 인식 학습 테스트 시작 (단위: ms, mV, pA, nS, pF) =====")
    start_time = time.time()

    # 2. 네트워크 구성
    N_INPUT = 10
    N_OUTPUT = 1
    N_NEURONS = N_INPUT + N_OUTPUT

    net = CuPyNetwork(dt=DT)

    # 뉴런 파라미터 (기본값 사용)
    soma_p = SomaParams()
    ais_p = AISParams()
    coup_p = CouplingParams()

    net.add_neurons(num_neurons=N_NEURONS, soma_p=soma_p, ais_p=ais_p, coup_p=coup_p)
    print(f"{N_NEURONS}개의 뉴런 생성 완료.")

    # 시냅스 파라미터
    syn_p = SynapseParams(
        g_max=1.0,      # nS
        A_minus=0.022,  # A_plus(0.02)보다 약간 크게 설정하여 안정성 확보
        eta_bcm=0.0     # STDP 효과만 보기 위해 BCM 비활성화
    )

    # 연결 설정: 모든 입력 뉴런 -> 1개의 출력 뉴런
    pre_indices = list(range(N_INPUT))
    post_indices = [N_INPUT] * N_INPUT
    np.random.seed(42)
    initial_weights = np.random.uniform(0.1, 0.3, size=N_INPUT)
    net.connect(pre_indices=pre_indices, post_indices=post_indices, syn_p=syn_p, w_init=initial_weights)
    print("입력 뉴런 그룹 -> 출력 뉴런 시냅스 연결 완료.")

    # 3. 학습 설정
    TARGET_NEURONS = [2, 5, 8] # 이 뉴런 그룹과 출력 뉴런을 함께 활성화
    STIM_CURRENT = 350.0 # pA, 입력 뉴런 자극 전류
    TEACHER_CURRENT = 350.0 # pA, 출력 뉴런 교사 신호 전류

    # 스파이크 및 가중치 모니터링 설정
    net.spike_monitors = {i: [] for i in range(N_NEURONS)}
    weight_history = [initial_weights.copy()]
    time_history = [0.0]

    # 4. 학습 루프 (시뮬레이션 실행)
    num_steps = int(T_SIM / DT)
    print(f"학습을 시작합니다 (총 시뮬레이션 시간: {T_SIM/1000:.1f}초).")

    for i in range(num_steps):
        # 항상 목표 뉴런 그룹과 출력 뉴런에 전류 주입
        net.neuron_params['soma_I_ext'].fill(0.0)
        for nid in TARGET_NEURONS:
            net.neuron_params['soma_I_ext'][nid] = STIM_CURRENT
        net.neuron_params['soma_I_ext'][N_INPUT] = TEACHER_CURRENT # Teacher forcing

        net.step()

        # 100ms 마다 가중치 기록
        if (i + 1) % int(100 / DT) == 0:
            current_weights = cp.asnumpy(net.synapse_states['syn_w'])
            weight_history.append(current_weights)
            time_history.append(net.time)
            weight_change = np.linalg.norm(current_weights - weight_history[-2])
            print(f"  T={net.time/1000:.1f}s, 가중치 변화량: {weight_change:.4f}")

    print("학습 완료.")

    # 5. 최종 결과 시각화
    final_weights = cp.asnumpy(net.synapse_states['syn_w'])
    weight_history = np.array(weight_history)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # 가중치 변화 과정 플롯
    ax1 = fig.add_subplot(gs[0])
    for i in range(N_INPUT):
        color = '#ff7f0e' if i in TARGET_NEURONS else '#1f77b4'
        ax1.plot(time_history, weight_history[:, i], color=color, alpha=0.8)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Synaptic Weight")
    ax1.set_title("Synaptic Weight Evolution Over Time")
    ax1.grid(True, alpha=0.2)

    # 최종 시냅스 가중치 플롯
    ax2 = fig.add_subplot(gs[1])
    colors = ['#ff7f0e' if i in TARGET_NEURONS else '#1f77b4' for i in range(N_INPUT)]
    ax2.bar(range(N_INPUT), final_weights, color=colors, label='Target Neurons')
    ax2.bar(range(N_INPUT), initial_weights, color='gray', alpha=0.5, label='Initial Weights')
    ax2.set_xticks(range(N_INPUT))
    ax2.set_xlabel("Input Neuron Index")
    ax2.set_ylabel("Synaptic Weight")
    ax2.set_title("Final Synaptic Weights")
    ax2.legend()

    # 스파이크 래스터 플롯
    ax3 = fig.add_subplot(gs[2])
    spike_data = net.get_spike_times()
    for i in range(N_NEURONS):
        if spike_data.get(i):
            is_target_input = i < N_INPUT and i in TARGET_NEURONS
            is_other_input = i < N_INPUT and i not in TARGET_NEURONS
            is_output = i == N_INPUT
            
            color = '#ff7f0e' if is_target_input else \
                    '#d62728' if is_output else '#1f77b4'
            ax3.eventplot(spike_data[i], linelengths=0.8, lineoffsets=i, color=color)

    ax3.set_yticks(range(N_NEURONS))
    ax3.set_yticklabels([f'In {i}' for i in range(N_INPUT)] + ['Out 0'])
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Neuron Index")
    ax3.set_title("Spike Raster Plot")
    ax3.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig("pattern_recognition_test_result.png", dpi=300)
    plt.show()

    end_time = time.time()
    print(f"총 실행 시간: {end_time - start_time:.3f}초")

if __name__ == '__main__':
    run_pattern_recognition_test()