import numpy as np

class Stimulus:
    """외부 자극/환경 인터페이스"""

    def __init__(self, N: int):
        self.N = N
        self.inputs = np.zeros(N, dtype=np.float32)

    def inject_current(self, neuron_id: int, I: float):
        self.inputs[neuron_id] += I

    def reset(self):
        self.inputs.fill(0.0)
