
import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

# 프로젝트 루트를 경로에 추가하여 src 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.neuro_pyramidal import (
    CuPyNetwork,
    SomaParams,
    AISParams,
    CouplingParams,
    SynapseParams
)

def run_pattern_recognition_test():
    """
    STDP를 이용한 스파이크 시간 패턴 인식 학습 테스트.
    - 입력 뉴런 그룹이 특정 시간 패턴으로 스파이크할 때만 출력 뉴런이 반응하도록 학습.
    """
    # 1. 시뮬레이션 파라미터
    T_SIM = 20.0  # 총 시뮬레이션 시간 (초) - 학습을 위해 더 길게 설정
    DT = 1e-4     # 시간 해상도

    print("===== STDP 패턴 인식 학습 테스트 시작 =====")
    start_time = time.time()

    # 2. 네트워크 구성
    N_INPUT = 10
    N_OUTPUT = 1
    N_NEURONS = N_INPUT + N_OUTPUT

    net = CuPyNetwork(dt=DT)

    # 뉴런 파라미터 (모든 뉴런에 동일하게 적용)
    soma_p = SomaParams(I_ext=0.0)
    ais_p = AISParams()
    coup_p = CouplingParams()

    # 입력 뉴런 그룹 추가
    net.add_neurons(num_neurons=N_INPUT, soma_p=soma_p, ais_p=ais_p, coup_p=coup_p)
    # 출력 뉴런 추가
    net.add_neurons(num_neurons=N_OUTPUT, soma_p=soma_p, ais_p=ais_p, coup_p=coup_p)

    print(f"{N_INPUT}개의 입력 뉴런과 {N_OUTPUT}개의 출력 뉴런 생성 완료.")

    # 시냅스 파라미터 (STDP 활성화, BCM은 비활성화)
    syn_p = SynapseParams(
        g_max=250.0e-9,  # 시냅스 최대 전도도를 최종 상향
        A_plus=0.4,      # 학습률 대폭 상향 (기존 0.01)
        A_minus=0.42,    # 학습률 대폭 상향 (기존 0.0105)
        tau_pre=20e-3,
        tau_post=20e-3,
        eta_bcm=0.0, # STDP 효과만 보기 위해 BCM 비활성화
        w_max=1.0    # 가중치 최대값을 1.0으로 제한
    )

    # 연결 설정: 모든 입력 뉴런 -> 1개의 출력 뉴런
    pre_indices = list(range(N_INPUT))
    post_indices = [N_INPUT] * N_INPUT # 모든 시냅스가 N_INPUT 인덱스(첫 출력 뉴런)를 가리킴

    # 초기 가중치는 0.1 ~ 0.3 사이의 무작위 값으로 설정
    np.random.seed(42) # 결과 재현을 위한 시드 설정
    initial_weights = np.random.uniform(0.1, 0.3, size=N_INPUT)

    net.connect(pre_indices=pre_indices, post_indices=post_indices, syn_p=syn_p, w_init=initial_weights)
    print("입력 뉴런 그룹 -> 출력 뉴런 시냅스 연결 완료.")

    # 스파이크 모니터링 설정
    net.spike_monitors = {i: [] for i in range(N_NEURONS)}

    # 3. 스파이크 패턴 정의
    TARGET_NEURONS = [2, 5, 8]
    PATTERN_INTERVAL = 15e-3 # 15ms

    def generate_spike_times(time_offset, is_target_pattern):
        """특정 시간(time_offset)에 목표 패턴 또는 무작위 패턴을 생성한다."""
        spike_times = {}
        if is_target_pattern:
            # 목표 패턴: 뉴런 2, 5, 8이 15ms 간격으로 순차적 스파이크
            for i, neuron_idx in enumerate(TARGET_NEURONS):
                spike_times[neuron_idx] = time_offset + i * PATTERN_INTERVAL
        else:
            # 무작위 패턴: 3개의 뉴런을 무작위로 선택하여 무작위 시간 간격으로 스파이크
            num_spikes = 3
            random_neurons = np.random.choice(N_INPUT, num_spikes, replace=False)
            random_intervals = np.random.uniform(5e-3, 25e-3, size=num_spikes)
            current_time = time_offset
            for i in range(num_spikes):
                spike_times[random_neurons[i]] = current_time
                current_time += random_intervals[i]
        return spike_times

    # 4. 학습 루프 (시뮬레이션 실행)
    # 학습 파라미터
    N_TRIALS = 200  # 총 학습 횟수
    TRIAL_DURATION = 100e-3 # 100ms 마다 패턴 제시
    TARGET_PROB = 0.5 # 목표 패턴이 나타날 확률
    STIM_CURRENT = 25000e-12 # 입력 뉴런 자극 전류
    TEACHER_CURRENT = 25000e-12 # 출력 뉴런 교사 신호 전류 (입력과 동일하게 상향)
    STIM_DURATION_STEPS = int(1e-3 / DT)

    num_steps = int(T_SIM / DT)
    last_weights = initial_weights.copy()

    print(f"{N_TRIALS}회의 학습을 시작합니다 (총 시뮬레이션 시간: {T_SIM}초).")

    spike_schedule = {} # 루프 시작 전 스파이크 계획표 명시적 초기화

    for i in range(num_steps):
        current_t = i * DT

        # 현재 스텝에서 주입할 전류 초기화
        net.neuron_params['soma_I_ext'].fill(0.0)

        # 매 TRIAL_DURATION 마다 새로운 패턴 생성 및 적용
        if i % int(TRIAL_DURATION / DT) == 0:
            is_target = np.random.rand() < TARGET_PROB
            pattern_start_time = current_t + 10e-3 # 10ms 후 패턴 시작
            spike_schedule = generate_spike_times(pattern_start_time, is_target)

            # 교사 신호 스케줄링 (목표 패턴일 경우에만)
            if is_target:
                teacher_spike_time = pattern_start_time + (len(TARGET_NEURONS)) * PATTERN_INTERVAL
                spike_schedule[N_INPUT] = teacher_spike_time # 출력 뉴런(N_INPUT 인덱스)에 교사 신호

        # 스케줄에 따라 뉴런에 전류 주입
        for neuron_idx, spike_time in spike_schedule.items():
            if abs(current_t - spike_time) < (STIM_DURATION_STEPS * DT / 2):
                current = TEACHER_CURRENT if neuron_idx == N_INPUT else STIM_CURRENT
                net.neuron_params['soma_I_ext'][neuron_idx] = current

        # 네트워크 1스텝 진행
        net.step()

        # 1초마다 중간 가중치 변화 출력
        if i % int(1.0 / DT) == 0 and i > 0:
            current_weights = cp.asnumpy(net.synapse_states['syn_w'])
            weight_change = np.linalg.norm(current_weights - last_weights)
            print(f"  T={current_t:.1f}s, 가중치 변화량: {weight_change:.4f}")
            last_weights = current_weights

    print("학습 완료. 최종 검증을 시작합니다...")

    # 5. 학습 결과 검증
    # 먼저, 더 이상의 학습이 일어나지 않도록 STDP 파라미터를 0으로 설정
    net.synapse_params['syn_A_plus'].fill(0.0)
    net.synapse_params['syn_A_minus'].fill(0.0)

    # 테스트 1: 목표 패턴에 반응하는지 확인
    print("  - 테스트 1: 목표 패턴 입력")
    spike_schedule = generate_spike_times(current_t + 50e-3, is_target_pattern=True)
    # 교사 신호 없이 스스로 반응하는지 확인
    output_spikes_before_test = len(net.spike_monitors[N_INPUT])

    for i in range(int(200e-3 / DT)): # 200ms 동안 테스트 진행
        current_t += DT
        net.neuron_params['soma_I_ext'].fill(0.0)
        for neuron_idx, spike_time in spike_schedule.items():
            if abs(current_t - spike_time) < (STIM_DURATION_STEPS * DT / 2):
                net.neuron_params['soma_I_ext'][neuron_idx] = STIM_CURRENT
        net.step()

    output_spikes_after_test = len(net.spike_monitors[N_INPUT])
    target_test_success = output_spikes_after_test > output_spikes_before_test
    print(f"    -> 출력 뉴런 스파이크 발생: {'성공' if target_test_success else '실패'}")

    # 테스트 2: 무작위 패턴에 반응하지 않는지 확인
    print("  - 테스트 2: 무작위 패턴 입력")
    spike_schedule = generate_spike_times(current_t + 50e-3, is_target_pattern=False)
    output_spikes_before_test = len(net.spike_monitors[N_INPUT])

    for i in range(int(200e-3 / DT)): # 200ms 동안 테스트 진행
        current_t += DT
        net.neuron_params['soma_I_ext'].fill(0.0)
        for neuron_idx, spike_time in spike_schedule.items():
            if abs(current_t - spike_time) < (STIM_DURATION_STEPS * DT / 2):
                net.neuron_params['soma_I_ext'][neuron_idx] = STIM_CURRENT
        net.step()
    
    output_spikes_after_test = len(net.spike_monitors[N_INPUT])
    random_test_success = output_spikes_after_test == output_spikes_before_test
    print(f"    -> 출력 뉴런 스파이크 없음: {'성공' if random_test_success else '실패'}")

    # 6. 최종 결과 시각화
    final_weights = cp.asnumpy(net.synapse_states['syn_w'])

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    # 시냅스 가중치 플롯
    ax1 = fig.add_subplot(gs[0])
    colors = ['#ff7f0e' if i in TARGET_NEURONS else '#1f77b4' for i in range(N_INPUT)]
    ax1.bar(range(N_INPUT), final_weights, color=colors)
    ax1.set_xticks(range(N_INPUT))
    ax1.set_xlabel("Input Neuron Index")
    ax1.set_ylabel("Synaptic Weight")
    ax1.set_title("Final Synaptic Weights after Training")
    ax1.axhline(np.mean(initial_weights), color='gray', linestyle='--', label='Initial Avg. Weight')
    ax1.legend()

    # 스파이크 래스터 플롯
    ax2 = fig.add_subplot(gs[1])
    spike_data = net.get_spike_times()
    for i in range(N_NEURONS):
        if spike_data.get(i):
            color = '#d62728' if i == N_INPUT else colors[i] if i < N_INPUT else 'gray'
            label = 'Output' if i == N_INPUT else f'Input {i}'
            ax2.eventplot(spike_data[i], linelengths=0.8, lineoffsets=i, color=color, label=label)
    ax2.set_yticks(range(N_NEURONS))
    ax2.set_yticklabels([f'In {i}' for i in range(N_INPUT)] + ['Out 0'])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neuron Index")
    ax2.set_title("Spike Raster Plot (During Verification)")
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig("pattern_recognition_test_result.png", dpi=300)
    plt.show()





    end_time = time.time()
    print(f"총 실행 시간: {end_time - start_time:.3f}초")


if __name__ == '__main__':
    run_pattern_recognition_test()
