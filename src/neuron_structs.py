import numpy as np

class NeuronState:
    """뉴런 상태 (SoA 방식)"""

    def __init__(self, N: int):
        self.N = N
        self.v = np.full(N, -65.0, dtype=np.float32)  # membrane potential
        self.m = np.zeros(N, dtype=np.float32)
        self.h = np.zeros(N, dtype=np.float32)
        self.n = np.zeros(N, dtype=np.float32)
        self.spikes = np.zeros(N, dtype=np.int32)
        self.I_syn = np.zeros(N, dtype=np.float32)

    def reset(self):
        self.v.fill(-65.0)
        self.m.fill(0.0)
        self.h.fill(0.0)
        self.n.fill(0.0)
        self.spikes.fill(0)
        self.I_syn.fill(0.0)
