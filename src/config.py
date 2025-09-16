from dataclasses import dataclass

@dataclass
class SimConfig:
    """시뮬레이션 전역 파라미터"""

    # Simulation parameters
    dt: float = 0.025
    T: int = 1000   # 총 step 수

    # Neuron population
    N_neurons: int = 100
    delay_len: int = 16

    # Synapse population
    N_synapses: int = 1000

    # Hodgkin-Huxley constants
    g_na: float = 120.0
    g_k: float = 36.0
    g_leak: float = 0.3
    E_na: float = 50.0
    E_k: float = -77.0
    E_leak: float = -54.387
    Cm: float = 1.0

    # Plasticity parameters
    A_plus: float = 0.01
    A_minus: float = -0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0

    @property
    def sim_consts(self):
        """GPU 커널로 전달할 상수 struct 흉내 (지금은 딕셔너리)"""
        return {
            "dt": self.dt,
            "g_na": self.g_na,
            "g_k": self.g_k,
            "g_leak": self.g_leak,
            "E_na": self.E_na,
            "E_k": self.E_k,
            "E_leak": self.E_leak,
            "Cm": self.Cm,
        }
