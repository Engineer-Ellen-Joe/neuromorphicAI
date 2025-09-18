import numpy as np

class Logger:
    """시뮬레이션 결과 기록 및 체크포인트"""

    def __init__(self):
        self.voltages = []
        self.spikes = []

    def record(self, v, spikes):
        self.voltages.append(v.copy())
        self.spikes.append(spikes.copy())

    def save(self, filename: str):
        np.savez(filename, voltages=self.voltages, spikes=self.spikes)
