"""
PyramidalLayer의 경쟁 학습 및 뉴런 전문화(specialization)를 검증합니다.

이 스크립트는 단일 PyramidalLayer를 시뮬레이션하며, 외부 제어 루프를 통해
'승자 독식(Winner-take-all)'과 유사한 경쟁 환경을 구현합니다.

- 시뮬레이션 루프는 각 스텝에서 뉴런의 발화를 감지합니다.
- 뉴런이 발화하면, 다음 스텝에서 다른 모든 뉴런에게 '억제성 전류'를
  주입하여 경쟁을 유도합니다.
- 이 과정을 통해, 초기에는 비슷했던 뉴런들이 점차 특정 입력 패턴에만
  선택적으로 반응하는 '전문가'로 분화하는지 확인합니다.
- 학습 완료 후, 각 뉴런이 어떤 패턴에 더 강하게 연결되었는지 시각화하여
  역할 분담이 성공적으로 이루어졌는지 검증합니다.
"""

import sys
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pyramidal_layer import PyramidalLayer, DTYPE

# --- 기본 설정 ---
GPU_AVAILABLE = cp.cuda.is_available()
try:
    plt.rcParams['font.family'] = 'Noto Sans KR'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("Noto Sans KR 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    pass

def _to_numpy(array):
    """CuPy 배열을 NumPy 배열로 변환합니다."""
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)

def define_patterns(num_afferents, dt):
    """학습에 사용할 두 개의 시공간 패턴을 정의합니다."""
    # 패턴을 구성하는 입력 뉴런의 인덱스
    pattern_A_afferents = np.arange(0, num_afferents // 2)
    pattern_B_afferents = np.arange(num_afferents // 2, num_afferents)
    
    # 각 패턴 내 스파이크 사이의 시간 간격 (ms)
    pattern_interval_ms = 10
    pattern_interval_steps = int(pattern_interval_ms / dt)

    def get_spike_sequence(afferent_indices):
        spikes = []
        for i, afferent_idx in enumerate(afferent_indices):
            time_step = i * pattern_interval_steps
            spikes.append((time_step, afferent_idx))
        return spikes

    pattern_A_spikes = get_spike_sequence(pattern_A_afferents)
    pattern_B_spikes = get_spike_sequence(pattern_B_afferents)
    
    # 패턴의 총 길이 (가장 긴 패턴 기준)
    pattern_duration_steps = max(len(pattern_A_spikes), len(pattern_B_spikes)) * pattern_interval_steps
    
    return pattern_A_spikes, pattern_B_spikes, pattern_duration_steps, pattern_A_afferents, pattern_B_afferents

def test_competition_and_specialization():
    """PyramidalLayer의 경쟁 학습 및 전문화 과정을 테스트합니다."""
    if not GPU_AVAILABLE:
        print("경쟁 학습 테스트: CUDA 장치가 없어 건너뜁니다.")
        return

    # --- 1. 환경 설정 ---
    print("--- 1. 환경 설정 시작 ---")
    np.random.seed(42)
    cp.random.seed(42)

    num_neurons = 20
    num_afferents = 10
    dt = 0.1  # ms

    # 패턴 정의
    pA_spikes, pB_spikes, p_steps, pA_aff, pB_aff = define_patterns(num_afferents, dt)
    
    # 편향된 초기 가중치 생성
    # 뉴런의 절반은 A패턴에, 나머지 절반은 B패턴에 초기 선호도를 갖도록 설정
    weights = cp.random.uniform(0.2, 0.5, size=(num_neurons, num_afferents)).astype(DTYPE)
    num_group_A = num_neurons // 2
    # 그룹 A 뉴런들 (0-9)은 패턴 A 입력(pA_aff)에 강한 초기 연결을 가짐
    weights[0:num_group_A, pA_aff] += 0.3
    # 그룹 B 뉴런들 (10-19)은 패턴 B 입력(pB_aff)에 강한 초기 연결을 가짐
    weights[num_group_A:, pB_aff] += 0.3

    # 레이어 초기화
    layer = PyramidalLayer(
        num_neurons=num_neurons,
        num_afferents=num_afferents,
        num_branches=1, # 이 테스트에서는 사용 안함
        dt=dt,
        membrane_time_constant=10.0,
        ais_threshold=-55.0,
        reset_potential=-70.0,
        leak_potential=-70.0,
        refractory_period=5.0,
        input_learning_rate=0.002, # 학습률 감소
        initial_input_weights=weights # 편향된 가중치 사용
    )

    # --- 2. 경쟁 학습 실행 ---
    print("--- 2. 경쟁 학습 시작 ---")
    trials = 200
    pattern_choices = [pA_spikes, pB_spikes]
    
    base_current = 0.05  # nA, 기본 전류 감소
    inhibition_current = -0.5 # nA
    
    for trial in range(trials):
        pattern_spikes = pattern_choices[np.random.randint(len(pattern_choices))]
        layer.reset_state()
        
        # 외부에서 제어되는 억제 전류
        inhibition = cp.zeros(num_neurons, dtype=DTYPE)

        for step in range(p_steps):
            # 현재 스텝에 해당하는 입력 스파이크 생성
            presynaptic_spikes = cp.zeros(num_afferents, dtype=DTYPE)
            for t, afferent_idx in pattern_spikes:
                if step == t:
                    presynaptic_spikes[afferent_idx] = 1.0
            
            # 기본 전류와 억제 전류를 합산하여 외부 전류로 주입
            external_currents = base_current + inhibition
            
            # 레이어 상태 업데이트
            result = layer.step(presynaptic_spikes, external_currents=external_currents)
            
            # --- 외부 제어 루프 (측면 억제) ---
            # 발화한 뉴런이 있는지 확인
            spiking_neurons = result.axon_spikes > 0
            if cp.any(spiking_neurons):
                # 다음 스텝을 위해 억제 전류를 새로 계산
                inhibition = cp.full(num_neurons, inhibition_current, dtype=DTYPE)
                # 발화한 모든 뉴런(승자들)은 억제에서 제외
                inhibition[spiking_neurons] = 0.0
            else:
                # 발화한 뉴런이 없으면 억제 해제
                inhibition.fill(0.0)

        if (trial + 1) % 20 == 0:
            print(f"  Trial {trial+1}/{trials} 완료...")

    print("--- 2. 경쟁 학습 완료 ---")

    # --- 3. 결과 검증 및 시각화 ---
    print("--- 3. 결과 검증 시작 ---")
    
    final_weights = _to_numpy(layer.input_weights)
    
    # 각 뉴런이 A 패턴과 B 패턴의 입력에 대해 얼마나 강하게 연결되었는지 계산
    affinity_A = final_weights[:, pA_aff].mean(axis=1)
    affinity_B = final_weights[:, pB_aff].mean(axis=1)

    plt.figure(figsize=(10, 8))
    plt.scatter(affinity_A, affinity_B, alpha=0.7)
    
    # 각 점에 뉴런 번호 표시
    for i in range(num_neurons):
        plt.text(affinity_A[i], affinity_B[i], str(i), fontsize=9)

    plt.title("학습 후 뉴런 전문화 결과", fontsize=16)
    plt.xlabel("패턴 A 선호도 (평균 가중치)", fontsize=12)
    plt.ylabel("패턴 B 선호도 (평균 가중치)", fontsize=12)
    plt.grid(True)
    plt.axline((0, 0), slope=1, color='r', linestyle='--', label="경계선")
    plt.legend()
    
    output_filename = "network_competition_result.png"
    plt.savefig(output_filename)
    print(f"학습 결과 그래프를 '{output_filename}' 파일로 저장했습니다.")
    plt.close()

if __name__ == "__main__":
    test_competition_and_specialization()
