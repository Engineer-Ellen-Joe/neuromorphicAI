"""
PyramidalNeuron의 시공간 패턴 인식 학습 능력을 검증합니다.

이 스크립트는 STDP 규칙을 사용하여 특정 스파이크 시퀀스(패턴)를 학습하는
PyramidalNeuron 모델을 시뮬레이션합니다. 학습 전후로 뉴런이 타겟 패턴과
노이즈 패턴에 어떻게 다르게 반응하는지, 그리고 시냅스 가중치가 어떻게
변화하는지를 시각화하여 보여줍니다.

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

# GPU 사용 가능 여부 확인
try:
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    GPU_AVAILABLE = False

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Noto Sans KR'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("Noto Sans KR 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    pass

def _to_numpy(array):
    """CuPy 배열을 NumPy 배열로 변환합니다."""
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)

def run_pattern_recognition_experiment():
    """
    STDP를 이용한 스파이크 패턴 인식 실험을 수행합니다.
    """
    if not GPU_AVAILABLE:
        print("Pattern Recognition Test: CUDA 장치가 없어 건너뜁니다.")
        return

    print("패턴 인식 학습 실험 시작...")

    # --- 1. 파라미터 설정 ---
    # test_pyramidal_neuron_pytest.py의 BASE_PARAMS를 참고하여 설정
    num_afferents = 20
    dt = 1e-4  # 더 정밀한 시간 제어를 위해 dt를 작게 설정

    neuron = PyramidalNeuron(
        num_afferents=num_afferents,
        num_branches=1, # 간단한 출력을 위해 branch는 1개로 설정
        dt=dt,
        membrane_time_constant=12e-3,
        membrane_capacitance=180e-12,
        leak_potential=-70e-3,
        reset_potential=-68e-3,
        ais_threshold=-61e-3,
        refractory_period=1e-3,
        input_learning_rate=0.015,
        output_learning_rate=0.0, # 입력 가중치 학습에만 집중
        stdp_a_plus=0.01,
        stdp_a_minus=0.0105,
        stdp_tau_pre=12e-3,
        stdp_tau_post=12e-3,
        initial_input_weights=np.full(num_afferents, 0.4),
        input_weight_bounds=(0.0, 1.0),
        random_state=42,
    )

    # --- 2. 타겟 패턴 및 노이즈 정의 ---
    target_afferents = [3, 8, 14]
    pattern_interval_ms = 10
    pattern_interval_steps = int(pattern_interval_ms * 1e-3 / dt)

    # 타겟 패턴: [t, t+10ms, t+20ms]에 걸쳐 특정 뉴런들이 순차적으로 발화
    target_pattern_spikes = []
    for i, afferent_idx in enumerate(target_afferents):
        time_step = i * pattern_interval_steps
        target_pattern_spikes.append((time_step, afferent_idx))
    
    pattern_duration_steps = len(target_afferents) * pattern_interval_steps

    # --- 3. 학습 과정 ---
    print("학습 시작...")
    epochs = 150
    presentation_interval_steps = int(100e-3 / dt) # 패턴 제시 간격
    
    weight_history = [ _to_numpy(neuron.input_weights).copy() ]
    
    # 학습을 위한 외부 전류 (뉴런이 패턴에 반응하여 발화하도록 돕는 역할)
    training_current = 200e-12 

    for epoch in range(epochs):
        # 뉴런 상태 초기화 (매 에포크마다)
        neuron.reset_state()
        
        # 타겟 패턴 제시
        for step in range(pattern_duration_steps):
            presynaptic_spikes = np.zeros(num_afferents)
            for t, afferent_idx in target_pattern_spikes:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            
            neuron.step(presynaptic_spikes, external_current=training_current)

        # 패턴 제시 후 휴지기
        for _ in range(presentation_interval_steps - pattern_duration_steps):
            neuron.step(np.zeros(num_afferents), external_current=0.0)

        # 가중치 기록
        weight_history.append(_to_numpy(neuron.input_weights).copy())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} 완료...")

    print("학습 완료.")
    final_weights = _to_numpy(neuron.input_weights)
    weight_history = np.array(weight_history)

    # --- 4. 검증 과정 ---
    print("학습 후 검증 시작...")
    
    def run_validation(pattern_type, spikes_sequence, duration_steps):
        neuron.reset_state()
        
        # 학습된 가중치 적용 (learning_rate=0으로 설정하여 가중치 동결)
        neuron.input_learning_rate_gpu = cp.float64(0.0)
        neuron.input_weights = cp.asarray(final_weights)

        potentials = []
        spikes = []
        
        # 검증 시에는 약간의 외부 전류만 인가
        validation_current = 80e-12

        for step in range(duration_steps):
            presynaptic_spikes = np.zeros(num_afferents)
            for t, afferent_idx in spikes_sequence:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            
            result = neuron.step(presynaptic_spikes, external_current=validation_current)
            potentials.append(result.soma_potential)
            spikes.append(result.axon_spike)
        
        return potentials, spikes

    # 4.1. 타겟 패턴에 대한 반응 검증
    target_potentials, target_spikes = run_validation(
        "Target Pattern", target_pattern_spikes, pattern_duration_steps
    )

    # 4.2. 노이즈 패턴에 대한 반응 검증
    noise_afferents = np.random.choice(num_afferents, len(target_afferents), replace=False)
    noise_pattern_spikes = []
    for i, afferent_idx in enumerate(noise_afferents):
        time_step = i * pattern_interval_steps
        noise_pattern_spikes.append((time_step, afferent_idx))

    noise_potentials, noise_spikes = run_validation(
        "Noise Pattern", noise_pattern_spikes, pattern_duration_steps
    )

    # --- 5. 시각화 ---
    print("결과 시각화...")
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle("STDP 기반 패턴 인식 학습 결과", fontsize=16)

    # 5.1. 가중치 변화 시각화
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(num_afferents):
        color = 'r' if i in target_afferents else 'gray'
        alpha = 1.0 if i in target_afferents else 0.3
        linestyle = '-' if i in target_afferents else '--'
        ax1.plot(weight_history[:, i], color=color, alpha=alpha, linestyle=linestyle)
    
    ax1.set_title("시냅스 가중치 변화 과정")
    ax1.set_xlabel("학습 Epoch")
    ax1.set_ylabel("가중치 (Weight)")
    ax1.grid(True)
    ax1.legend([f"Target Afferent {i}" for i in target_afferents], loc='upper left')


    time_axis = np.arange(pattern_duration_steps) * dt * 1000 # ms 단위

    # 5.2. 타겟 패턴 반응 시각화
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_axis, target_potentials, label="막 전위 (mV)")
    spike_times = time_axis[np.array(target_spikes) > 0]
    ax2.plot(spike_times, np.full_like(spike_times, 0), 'ro', label="뉴런 스파이크")
    ax2.axhline(neuron.ais_threshold, color='r', linestyle='--', label="발화 임계값")
    ax2.set_title("학습 후 타겟 패턴에 대한 반응")
    ax2.set_xlabel("시간 (ms)")
    ax2.set_ylabel("전위 (mV)")
    ax2.legend()
    ax2.grid(True)

    # 5.3. 노이즈 패턴 반응 시각화
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(time_axis, noise_potentials, label="막 전위 (mV)")
    spike_times_noise = time_axis[np.array(noise_spikes) > 0]
    if len(spike_times_noise) > 0:
        ax3.plot(spike_times_noise, np.full_like(spike_times_noise, 0), 'ro', label="뉴런 스파이크")
    ax3.axhline(neuron.ais_threshold, color='r', linestyle='--', label="발화 임계값")
    ax3.set_title("학습 후 노이즈 패턴에 대한 반응")
    ax3.set_xlabel("시간 (ms)")
    ax3.set_ylabel("전위 (mV)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('C:\\Dev\\2025\\EllenProject\\test\\pattern_recognition_result.png')
    plt.show()


if __name__ == "__main__":
    run_pattern_recognition_experiment()
