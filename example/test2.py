import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import time

# neuro_pyramidal_cupy.py 파일에서 필요한 클래스들을 가져옵니다.
from src.neuro_pyramidal import (
    CuPyNetwork,
    SomaParams,
    AISParams,
    CouplingParams,
    SynapseParams
)

def run_test_simulation():
    """
    네트워크 시뮬레이션을 설정하고 실행한 뒤, 결과를 시각화하는 메인 함수입니다.
    """
    print("===== 신경망 시뮬레이션 테스트 시작 =====")

    # 1. 시뮬레이션 파라미터 설정
    T_SIM = 1.0  # 총 시뮬레이션 시간 (초)
    DT = 1e-4    # 시간 해상도 (초)
    N_NEURONS = 100 # 전체 뉴런 수
    P_CONNECT = 0.1 # 뉴런 간 무작위 연결 확률

    # 2. 네트워크 객체 생성
    net = CuPyNetwork(dt=DT)
    print(f"네트워크 생성 완료 (dt={DT}s)")

    # 3. 뉴런 파라미터 정의 및 네트워크에 추가
    # 모든 뉴런에 동일한 파라미터를 적용합니다.
    soma_p = SomaParams(I_ext=0.0) # 기본 외부 전류는 0으로 설정
    ais_p = AISParams()
    coup_p = CouplingParams()
    
    net.add_neurons(num_neurons=N_NEURONS, soma_p=soma_p, ais_p=ais_p, coup_p=coup_p)
    print(f"{N_NEURONS}개의 뉴런 추가 완료.")

    # 4. 일부 뉴런에 외부 자극(전류) 주입
    # 처음 10개의 뉴런에만 활동을 유발하기 위한 외부 전류를 설정합니다.
    # 이 전류는 네트워크에 초기 활동을 만들어냅니다.
    n_driven_neurons = 10
    driven_current = 400e-12 # 400 pA
    net.neuron_params['soma_I_ext'][:n_driven_neurons] = driven_current
    print(f"초기 뉴런 {n_driven_neurons}개에 {driven_current*1e12:.0f} pA 전류 주입 설정 완료.")

    # 5. 시냅스 파라미터 정의 및 뉴런 연결
    # 흥분성 시냅스 파라미터를 설정합니다.
    syn_p = SynapseParams(
        g_max=2.0e-9,      # 최대 시냅스 전도도
        tau_decay=5e-3,    # 시냅스 전류 감쇠 시간
        delay=1.5e-3       # 축삭돌기 전파 지연 시간
    )

    # 무작위 연결 생성
    pre_indices, post_indices = [], []
    for i in range(N_NEURONS):
        for j in range(N_NEURONS):
            # 자기 자신을 제외하고, P_CONNECT 확률로 연결
            if i != j and np.random.rand() < P_CONNECT:
                pre_indices.append(i)
                post_indices.append(j)
    
    net.connect(pre_indices, post_indices, syn_p, w_init=1.0)
    print(f"{len(pre_indices)}개의 시냅스 연결 완료 (연결 확률: {P_CONNECT}).")

    # 6. 시뮬레이션 실행
    # 처음 20개 뉴런의 스파이크를 모니터링합니다.
    neurons_to_monitor = list(range(20))
    
    start_time = time.time()
    net.run(T=T_SIM, monitor_spikes_for=neurons_to_monitor)
    end_time = time.time()
    
    print(f"시뮬레이션 실행 시간: {end_time - start_time:.2f}초")

    # 7. 결과 시각화
    spike_data = net.get_spike_times()

    if not any(spike_data.values()):
        print("\n경고: 시뮬레이션 동안 스파이크가 발생하지 않았습니다.")
        print("soma_I_ext 값을 높이거나 더 많은 뉴런에 전류를 주입해보세요.")
        return

    print("\n결과 시각화 중...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 래스터 플롯 생성
    raster_data = []
    for neuron_idx, spike_times in spike_data.items():
        if spike_times: # 스파이크가 있는 경우에만 추가
            raster_data.append(spike_times)
    
    colors = plt.cm.viridis(np.linspace(0.4, 1, len(raster_data)))
    
    ax.eventplot(
        raster_data, 
        linelengths=0.9, 
        lineoffsets=[idx for idx in spike_data if spike_data[idx]],
        colors=colors
    )

    ax.set_title(f'Raster Plot of Neuron Spikes (First {len(neurons_to_monitor)} Neurons)', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_yticks(neurons_to_monitor)
    ax.set_xlim(0, T_SIM)
    ax.set_ylim(-1, len(neurons_to_monitor))
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 플롯 저장 및 표시
    output_filename = "neuron_spike_raster_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"플롯이 '{output_filename}' 파일로 저장되었습니다.")
    plt.show()


if __name__ == '__main__':
    run_test_simulation()