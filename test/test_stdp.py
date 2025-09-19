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

def run_stdp_test(potentiation_test: bool = True):
    """
    STDP 학습 규칙을 검증하기 위한 시뮬레이션.
    - 뉴런 A (pre-synaptic) -> 뉴런 B (post-synaptic)
    - A와 B의 스파이크 시간 차이에 따라 시냅스 가중치가 변하는지 확인.

    Args:
        potentiation_test (bool): True이면 가중치 강화(LTP) 테스트, False이면 약화(LTD) 테스트.
    """
    if potentiation_test:
        print("===== STDP 가중치 강화 (LTP) 테스트 시작 =====")
        # A가 B보다 10ms 먼저 스파이크하도록 설정
        spike_time_A = 0.100  # 100ms
        spike_time_B = 0.110  # 110ms (A 이후)
        title = "STDP Potentiation (Pre-then-Post)"
    else:
        print("===== STDP 가중치 약화 (LTD) 테스트 시작 =====")
        # B가 A보다 10ms 먼저 스파이크하도록 설정
        spike_time_A = 0.110  # 110ms (B 이후)
        spike_time_B = 0.100  # 100ms
        title = "STDP Depression (Post-then-Pre)"

    # 1. 시뮬레이션 파라미터
    T_SIM = 0.25  # 총 시뮬레이션 시간 (초)
    DT = 1e-4     # 시간 해상도
    N_NEURONS = 2 # 뉴런 2개 (A, B)

    # 2. 네트워크 생성
    net = CuPyNetwork(dt=DT)

    # 3. 뉴런 추가
    # 외부 전류는 0으로 시작. 스파이크는 직접 유도할 것임.
    soma_p = SomaParams(I_ext=0.0)
    ais_p = AISParams()
    coup_p = CouplingParams()
    # 스파이크 모니터를 수동으로 활성화 (net.run()을 사용하지 않기 때문)
    net.spike_monitors = {0: [], 1: []}
    net.add_neurons(num_neurons=N_NEURONS, soma_p=soma_p, ais_p=ais_p, coup_p=coup_p)

    # 4. 시냅스 연결 (A -> B)
    # STDP 파라미터는 neuro_pyramidal.py의 기본값을 사용
    syn_p = SynapseParams(
        g_max=1.0e-9,
        A_plus=0.01,   # 가중치 변화량을 더 잘 보기 위해 test2.py보다 약간 높게 설정
        A_minus=0.0105, # A_plus와 거의 같지만 약간 높게 설정하여 균형을 맞춤
        tau_pre=20e-3,
        tau_post=20e-3,
    )
    initial_weight = 0.5
    net.connect(pre_indices=[0], post_indices=[1], syn_p=syn_p, w_init=initial_weight)
    print(f"뉴런 0 -> 뉴런 1 시냅스 연결 완료. 초기 가중치: {initial_weight}")

    # 5. 시뮬레이션 및 결과 기록
    num_steps = int(T_SIM / DT)
    weight_history = [initial_weight]
    time_points = [0]

    # 특정 시간에 스파이크를 유도하기 위한 외부 전류 펄스 설정
    # 매우 짧고 강한 전류를 주어 정확한 타이밍에 스파이크를 유발
    stim_current = 25000e-12 # 25000 pA (안정적인 스파이크를 유발하는 수준으로 조정)
    stim_duration_steps = int(1e-3 / DT) # 1ms 동안 자극

    print(f"시뮬레이션 시작: {T_SIM}초 동안 실행됩니다.")
    start_time = time.time()

    for i in range(num_steps):
        current_t = i * DT

        # 뉴런 A에 스파이크 유도
        if abs(current_t - spike_time_A) < (stim_duration_steps * DT / 2):
            net.neuron_params['soma_I_ext'][0] = stim_current
        else:
            net.neuron_params['soma_I_ext'][0] = 0.0

        # 뉴런 B에 스파이크 유도
        if abs(current_t - spike_time_B) < (stim_duration_steps * DT / 2):
            net.neuron_params['soma_I_ext'][1] = stim_current
        else:
            net.neuron_params['soma_I_ext'][1] = 0.0

        net.step()

        # 1ms 마다 시냅스 가중치 기록 (너무 자주 기록하면 성능 저하)
        if (i + 1) % int(1e-3 / DT) == 0:
            # CuPy 배열에서 값을 가져올 때는 .get() 사용
            current_weight = net.synapse_states['syn_w'][0].get()
            weight_history.append(current_weight)
            time_points.append(current_t)

    end_time = time.time()
    print(f"시뮬레이션 실행 시간: {end_time - start_time:.3f}초")
    final_weight = weight_history[-1]
    print(f"최종 시냅스 가중치: {final_weight:.4f}")

    # 6. 결과 시각화
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # 래스터 플롯
    spike_data = net.get_spike_times()
    ax1.eventplot(
        [spike_data.get(0, []), spike_data.get(1, [])],
        linelengths=0.9,
        lineoffsets=[0, 1],
        colors=['#33a02c', '#1f78b4']
    )
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Neuron Index', fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Neuron A (pre)', 'Neuron B (post)'])
    ax1.set_ylim([-0.5, 1.5])

    # 가중치 변화 플롯
    ax2.plot(time_points, weight_history, marker='o', linestyle='-', markersize=3, label='Synaptic Weight (A->B)')
    ax2.axhline(initial_weight, color='r', linestyle='--', label=f'Initial Weight ({initial_weight})')
    ax2.set_title("Synaptic Weight Change Over Time", fontsize=14)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 스파이크 발생 시간에 수직선 표시
    ax1.axvline(spike_time_A, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(spike_time_B, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(spike_time_A, color='gray', linestyle=':', alpha=0.7, label='Spike A')
    ax2.axvline(spike_time_B, color='gray', linestyle=':', alpha=0.7, label='Spike B')


    plt.tight_layout()
    output_filename = f"stdp_{{'potentiation' if potentiation_test else 'depression'}}_test.png"
    plt.savefig(output_filename, dpi=300)
    print(f"플롯이 '{output_filename}' 파일로 저장되었습니다.\n")
    plt.show()

    print("=== Spike monitors (CPU view) ===")
    for nid, times in net.get_spike_times().items():
        print(f"Neuron {nid}: spikes = {times}")

    print("=== Synapse final states (GPU arrays -> numpy) ===")
    print("syn_w:", float(cp.asnumpy(net.synapse_states['syn_w'])[0]))
    print("syn_pre_trace:", float(cp.asnumpy(net.synapse_states['syn_pre_trace'])[0]))
    print("syn_post_trace:", float(cp.asnumpy(net.synapse_states['syn_post_trace'])[0]))
    print("syn_r_pre:", float(cp.asnumpy(net.synapse_states['syn_r_pre'])[0]))
    print("syn_r_post:", float(cp.asnumpy(net.synapse_states['syn_r_post'])[0]))
    print("syn_theta_m:\n", float(cp.asnumpy(net.synapse_states['syn_theta_m'])[0]))

    final_weight = float(cp.asnumpy(net.synapse_states['syn_w'][0]))

    success = final_weight > initial_weight if potentiation_test else final_weight < initial_weight
    return success, final_weight

if __name__ == '__main__':
    # 1. 가중치 강화(Potentiation) 테스트 실행
    success, final_weight = run_stdp_test(True)
    print("LTP Test Result:", "Success" if success else "Failure", "| Final weight:", final_weight)

    # 2. 가중치 약화(Depression) 테스트 실행
    success, final_weight = run_stdp_test(False)
    print("LTD Test Result:", "Success" if success else "Failure", "| Final weight:", final_weight)
