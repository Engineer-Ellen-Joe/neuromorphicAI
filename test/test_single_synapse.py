
import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.neuro_pyramidal import (
    CuPyNetwork,
    SomaParams,
    AISParams,
    CouplingParams,
    SynapseParams
)

def run_single_synapse_test():
    """
    단일 시냅스가 최대 가중치(w=1.0)와 최대 전도도(g_max)를 가질 때
    출력 뉴런을 스파이크 시킬 수 있는지 확인하는 최종 테스트.
    """
    # 1. 시뮬레이션 파라미터
    T_SIM = 50e-3  # 50ms의 짧은 시간만 시뮬레이션
    DT = 1e-4
    N_INPUT = 1
    N_OUTPUT = 1
    N_NEURONS = N_INPUT + N_OUTPUT

    print("===== 단일 시냅스 효력 테스트 시작 =====")

    # 2. 네트워크 구성
    net = CuPyNetwork(dt=DT)
    soma_p = SomaParams()
    ais_p = AISParams()
    coup_p = CouplingParams()
    net.add_neurons(num_neurons=N_NEURONS, soma_p=soma_p, ais_p=ais_p, coup_p=coup_p)

    # 3. 시냅스 설정 (학습 기능 없이, 최대 힘으로 고정)
    syn_p = SynapseParams(
        g_max=250.0e-9,  # 이전 테스트에서 설정한 최대 전도도
        tau_decay=20e-3, # 이전 테스트에서 설정한 지속 시간
        A_plus=0.0,      # 학습 기능 비활성화
        A_minus=0.0,     # 학습 기능 비활성화
    )
    
    # 가중치를 1.0으로 고정
    initial_weight = 1.0

    net.connect(pre_indices=[0], post_indices=[1], syn_p=syn_p, w_init=initial_weight)
    print(f"단일 시냅스 연결 완료 (g_max={syn_p.g_max:.2e}, w={initial_weight:.1f})")

    # 4. 시뮬레이션
    net.spike_monitors = {0: [], 1: []}
    num_steps = int(T_SIM / DT)
    
    # 입력 뉴런(0번)을 10ms 시점에 스파이크 시킴
    input_spike_time = 10e-3
    stim_current = 25000e-12
    stim_duration_steps = int(1e-3 / DT)

    for i in range(num_steps):
        current_t = i * DT
        net.neuron_params['soma_I_ext'].fill(0.0)

        if abs(current_t - input_spike_time) < (stim_duration_steps * DT / 2):
            net.neuron_params['soma_I_ext'][0] = stim_current

        net.step()

    # 5. 결과 확인
    spike_data = net.get_spike_times()
    input_spikes = spike_data.get(0, [])
    output_spikes = spike_data.get(1, [])

    print(f"입력 뉴런 스파이크 시간: {input_spikes}")
    print(f"출력 뉴런 스파이크 시간: {output_spikes}")

    if len(input_spikes) > 0 and len(output_spikes) > 0:
        print("-> 결과: 성공! 단일 시냅스가 출력 뉴런을 스파이크 시켰습니다.")
        # 스파이크 지연 시간 계산
        delay = output_spikes[0] - input_spikes[0]
        print(f"   (입력 후 {delay*1000:.2f} ms 만에 반응)")
    else:
        print("-> 결과: 실패. 단일 시냅스로는 스파이크를 만들지 못했습니다.")
        print("   이는 커널의 시냅스 전류 계산/적용 부분에 근본적인 문제가 있음을 시사합니다.")

if __name__ == '__main__':
    run_single_synapse_test()

