import sys
from pathlib import Path
import cupy as cp
import numpy as np

# src 모듈 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import SimConfig
from neuron_structs import NeuronState
from gpu.gpu_utils import GPUUtils
from gpu.neuron_ops import NeuronOps
from cpu.logging import Logger
from cpu.visualization import Visualization

def main():
    # 1. 설정 로드
    config = SimConfig(N_neurons=1, T=200)

    # 2. CPU 쪽 뉴런 상태 초기화 (ground truth / 기록용)
    neuron_state = NeuronState(config.N_neurons)

    # 3. GPU 유틸 및 뉴런 연산자 초기화 (config 전달!)
    gpu_utils = GPUUtils()
    neuron_ops = NeuronOps(gpu_utils, config.N_neurons, config)

    # 4. 로거 & 시각화 준비
    logger = Logger()
    viz = Visualization()

    # 5. 시뮬레이션 루프
    for t in range(config.T):
        neuron_ops.step()                     # GPU에서 한 스텝 진행
        v = cp.asnumpy(neuron_ops.v)          # CPU로 결과 복사
        spikes = cp.asnumpy(neuron_ops.spikes)

        # 기록
        logger.record(v, spikes)

    # 6. 결과 저장 및 플로팅
    logger.save("single_neuron_results.npz")
    data = np.load("single_neuron_results.npz", allow_pickle=True)
    voltages = np.array(data["voltages"])
    spikes = np.array(data["spikes"])

    viz.plot_voltage(voltages, neuron_id=0)
    viz.plot_spike_raster(spikes)

if __name__ == "__main__":
    main()
