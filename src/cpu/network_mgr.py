from src.gpu.neuron_ops import NeuronOps
from src.gpu.synapse_ops import SynapseOps
from src.gpu.axon_ops import AxonOps
from src.gpu.plasticity_ops import PlasticityOps
from src.gpu.gpu_utils import GPUUtils

from src.config import SimConfig

class NetworkManager:
    def __init__(self, config: SimConfig):
        self.config = config
        self.gpu_utils = GPUUtils()

        # GPU 모듈 초기화
        self.neuron_ops = NeuronOps(self.gpu_utils, config.N_neurons)
        self.synapse_ops = SynapseOps(self.gpu_utils, config.N_neurons)
        self.axon_ops = AxonOps(self.gpu_utils, config.N_neurons, config.delay_len)
        self.plasticity_ops = PlasticityOps(self.gpu_utils, config.N_synapses)

        # TODO: topology 로딩 (CSR)
        # TODO: stimulus/inputs 연결

    def step(self, t: int):
        """한 스텝 시뮬레이션"""
        # 1. 뉴런 업데이트
        self.neuron_ops.step(self.config.sim_consts)

        # 2. 축삭 지연/라우팅 처리
        self.axon_ops.step_delay(self.neuron_ops.spikes, t)

        # 3. 시냅스 전류 업데이트
        self.synapse_ops.step()

        # 4. 플라스티시티 업데이트
        self.plasticity_ops.step(
            self.config.A_plus, self.config.A_minus,
            self.config.tau_plus, self.config.tau_minus
        )

    def run(self, T: int):
        """전체 시뮬레이션 실행"""
        for t in range(T):
            self.step(t)
