"""
PyramidalNeuron의 각 구성 요소를 시각적으로 디버깅하고 검증합니다.

이 스크립트는 PyramidalNeuron 모델의 주요 메커니즘을 시뮬레이션하고,
matplotlib을 사용하여 결과를 그래프로 시각화합니다. 각 함수는 뉴런의 특정 동작
(예: 막 전위 동역학, STDP, BCM)을 격리하여 테스트합니다.

실행 전, `matplotlib` 라이브러리가 설치되어 있어야 합니다:
pip install matplotlib
"""

import math
import sys, os
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트를 경로에 추가하여 src 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_neuron import PyramidalNeuron, StepResult

# GPU 사용 가능 여부 확인
try:
    _CUDA_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = _CUDA_DEVICE_COUNT > 0
except cp.cuda.runtime.CUDARuntimeError:
    GPU_AVAILABLE = False

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Noto Sans KR'
    # 마이너스 부호가 깨지는 문제 해결
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Noto Sans KR 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다. 오류: {e}")
    # 대체 폰트나 다른 처리
    pass

def _to_numpy(array):
    """CuPy 배열을 NumPy 배열로 변환합니다."""
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)

def _scalar(array) -> float:
    """배열에서 스칼라 값을 추출합니다."""
    return float(_to_numpy(array).reshape(-1)[0])

def debug_soma_dynamics_and_spiking():
    """
    막 전위, AIS 활성화 및 스파이크 발생을 디버깅합니다.
    일정 시간 동안 외부 전류를 주입하여 뉴런의 반응을 시뮬레이션하고,
    주요 상태 변수들의 변화를 그래프로 확인합니다.
    """
    if not GPU_AVAILABLE:
        print("Soma Dynamics Test: CUDA 장치가 없어 건너뜁니다.")
        return

    print("막 전위 및 스파이크 발생 디버깅 시작...")
    neuron = PyramidalNeuron(
        num_afferents=1,
        num_branches=1,
        dt=1e-4,
        membrane_time_constant=10e-3,
        ais_threshold=-55e-3,
        reset_potential=-70e-3,
        leak_potential=-70e-3,
        refractory_period=5e-3,
    )

    duration = 100e-3  # 시뮬레이션 시간
    num_steps = int(duration / neuron.dt)
    time = np.arange(num_steps) * neuron.dt

    # 전체 시뮬레이션 시간 동안 외부 전류 주입
    external_current_val = 500e-12  # 500 pA (연속 발화를 유도하기 위해 값 조정)
    external_current_stimulus = np.full(num_steps, external_current_val)

    # 시뮬레이션 데이터 기록
    potentials = np.zeros(num_steps)
    ais_activations = np.zeros(num_steps)
    spikes = np.zeros(num_steps)

    for i in range(num_steps):
        result = neuron.step(
            presynaptic_spikes=[0.0],
            external_current=external_current_stimulus[i]
        )
        potentials[i] = result.soma_potential
        ais_activations[i] = result.ais_activation
        spikes[i] = result.axon_spike

    # 결과 시각화
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("막 전위, AIS 활성화 및 스파이크 발생 디버깅", fontsize=16)

    # 1. 막 전위
    ax1.plot(time * 1000, potentials * 1000, label="Soma Potential (mV)")
    ax1.axhline(neuron.ais_threshold * 1000, color='r', linestyle='--', label="AIS Threshold (mV)")
    ax1.axhline(neuron.reset_potential * 1000, color='g', linestyle='--', label="Reset Potential (mV)")
    ax1.set_ylabel("전위 (mV)")
    ax1.legend()
    ax1.grid(True)

    # 2. AIS 활성화
    ax2.plot(time * 1000, ais_activations, label="AIS Activation", color='orange')
    ax2.axhline(neuron.ais_activation_gate, color='r', linestyle='--', label="Activation Gate")
    ax2.set_ylabel("활성화 수준")
    ax2.legend()
    ax2.grid(True)

    # 3. 스파이크 및 외부 자극
    ax3.plot(time * 1000, spikes, 'ro', markersize=8, label="Axon Spike")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time * 1000, external_current_stimulus * 1e12, 'b--', alpha=0.5, label="External Current (pA)")
    ax3.set_xlabel("시간 (ms)")
    ax3.set_ylabel("스파이크 (1=발생)")
    ax3_twin.set_ylabel("외부 전류 (pA)")
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("완료. 그래프를 확인하세요.")


def debug_stdp_weight_change():
    """
    STDP 규칙에 따른 입력 가중치 변화를 디버깅합니다.
    사전 시냅스 스파이크와 사후 시냅스 스파이크 간의 시간 차이를 조절하며
    가중치가 예상대로 (LTP 또는 LTD) 변화하는지 확인합니다.
    """
    if not GPU_AVAILABLE:
        print("STDP Test: CUDA 장치가 없어 건너뜁니다.")
        return

    print("STDP 가중치 변화 디버깅 시작...")
    
    # 시뮬레이션 파라미터
    tau_pre = 20e-3
    tau_post = 20e-3
    a_plus = 0.01
    a_minus = 0.012
    initial_weight = 0.5
    learning_rate = 0.1
    dt = 1e-3

    neuron = PyramidalNeuron(
        num_afferents=1,
        num_branches=1,
        dt=dt,
        initial_input_weights=[initial_weight],
        stdp_a_plus=a_plus,
        stdp_a_minus=a_minus,
        stdp_tau_pre=tau_pre,
        stdp_tau_post=tau_post,
        input_learning_rate=learning_rate,
        ais_threshold=1.0,  # 스파이크를 수동으로 제어하기 위해 비활성화
    )

    time_diffs = np.linspace(-50e-3, 50e-3, 51)
    weight_changes = []

    for dt_spike in time_diffs:
        neuron.reset_state()
        neuron.input_weights.fill(initial_weight)
        
        # 스파이크 타이밍 설정
        pre_spike_time = 50e-3
        post_spike_time = pre_spike_time + dt_spike

        # 시뮬레이션
        pre_spike_fired = False
        post_spike_fired = False
        
        # STDP는 스파이크 발생 시점의 trace 값에 따라 결정됨
        # pre-spike -> post-spike (LTP)
        if dt_spike > 0:
            # pre-spike 발생
            neuron.step([1.0]) 
            # post-spike 발생까지 대기
            wait_steps = int(dt_spike / dt) -1
            for _ in range(wait_steps):
                neuron.step([0.0])
            # post-spike를 강제로 발생시킴 (실제로는 soma 전위가 임계값을 넘어야 함)
            neuron.post_trace.fill(1.0) # post-spike 효과 모방
            neuron.step([0.0])

        # post-spike -> pre-spike (LTD)
        else:
            # post-spike 발생
            neuron.post_trace.fill(1.0)
            neuron.step([0.0])
            # pre-spike 발생까지 대기
            wait_steps = int(abs(dt_spike) / dt) -1
            for _ in range(wait_steps):
                neuron.step([0.0])
            # pre-spike 발생
            neuron.step([1.0])

        final_weight = _scalar(neuron.input_weights)
        weight_changes.append(final_weight - initial_weight)

    # 이론적인 STDP 곡선
    theoretical_dw_ltp = learning_rate * a_plus * np.exp(-time_diffs[time_diffs > 0] / tau_pre)
    theoretical_dw_ltd = -learning_rate * a_minus * np.exp(time_diffs[time_diffs < 0] / tau_post)

    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.suptitle("STDP 가중치 변화 디버깅", fontsize=16)
    plt.plot(time_diffs * 1000, weight_changes, 'o-', label="Simulated Δw")
    plt.plot(time_diffs[time_diffs > 0] * 1000, theoretical_dw_ltp, 'g--', label="Theoretical LTP")
    plt.plot(time_diffs[time_diffs < 0] * 1000, theoretical_dw_ltd, 'r--', label="Theoretical LTD")
    
    plt.axhline(0, color='k', linestyle='-')
    plt.axvline(0, color='k', linestyle='-')
    plt.xlabel("Δt = t_post - t_pre (ms)")
    plt.ylabel("가중치 변화 (Δw)")
    plt.title("STDP 학습 창")
    plt.legend()
    plt.grid(True)
    print("완료. 그래프를 확인하세요.")

def debug_bcm_ltp_dynamics():
    """
    BCM-style LTP에 따른 출력 가중치 변화를 디버깅합니다.
    지속적인 활동을 통해 BCM 임계값(theta)과 출력 가중치가 어떻게 변하는지 확인합니다.
    """
    if not GPU_AVAILABLE:
        print("BCM LTP Test: CUDA 장치가 없어 건너뜁니다.")
        return

    print("BCM-style LTP 동역학 디버깅 시작...")
    neuron = PyramidalNeuron(
        num_afferents=1,
        num_branches=1,
        dt=1e-3,
        initial_input_weights=[5e-9],
        initial_output_weights=[0.5],
        safety_factors=[1.0],
        axon_spike_current=10.0,
        conduction_threshold=0.5,
        branch_activity_tau=50e-3,
        bcm_tau=200e-3,
        output_learning_rate=0.2,
        input_learning_rate=0.0,
    )

    duration = 1.0  # 1초 시뮬레이션
    num_steps = int(duration / neuron.dt)
    time = np.arange(num_steps) * neuron.dt

    # 10Hz의 주기적인 스파이크 입력
    spike_train = np.zeros(num_steps)
    spike_interval = int(0.1 / neuron.dt)
    spike_train[::spike_interval] = 1.0

    # 데이터 기록
    output_weights = np.zeros(num_steps)
    thetas = np.zeros(num_steps)
    activities = np.zeros(num_steps)

    for i in range(num_steps):
        result = neuron.step([spike_train[i]])
        output_weights[i] = _scalar(neuron.output_weights)
        thetas[i] = _scalar(neuron.bcm_theta)
        activities[i] = _scalar(neuron.branch_activity_tau)

    # 결과 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("BCM-style LTP 동역학 디버깅", fontsize=16)

    # 1. 출력 가중치와 BCM 임계값(theta)
    ax1.plot(time, output_weights, label="Output Weight (w_out)")
    ax1.plot(time, thetas, label="BCM Threshold (θ)")
    ax1.set_ylabel("값")
    ax1.legend()
    ax1.grid(True)

    # 2. 가지 활성도 (Activity)
    ax2.plot(time, activities, label="Branch Activity Trace", color='purple')
    ax2.set_xlabel("시간 (s)")
    ax2.set_ylabel("활성도")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print("완료. 그래프를 확인하세요.")

if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("경고: CUDA 지원 GPU를 찾을 수 없습니다. 모든 테스트를 건너뜁니다.")
    else:
        # 각 디버깅 함수를 순차적으로 호출
        debug_soma_dynamics_and_spiking()
        debug_stdp_weight_change()
        debug_bcm_ltp_dynamics()

        # 모든 그래프를 화면에 표시
        plt.show()
