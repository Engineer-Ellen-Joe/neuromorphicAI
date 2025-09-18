import numpy as np

class SynapseState:
    """시냅스 상태 (CSR 기반)"""

    def __init__(self, N_syn: int, N_post: int):
        self.N_syn = N_syn
        self.N_post = N_post

        # CSR (compressed sparse row)
        self.conn_idx = np.zeros(N_syn, dtype=np.int32)     # pre neuron index
        self.conn_offset = np.zeros(N_post + 1, dtype=np.int32)

        # Synaptic weights & dynamics
        self.weights = np.random.normal(0.5, 0.1, N_syn).astype(np.float32)
        self.delays = np.ones(N_syn, dtype=np.int32)        # discrete delays
        self.p_release = np.ones(N_syn, dtype=np.float32)   # release probability

        # Short-term plasticity variables (Tsodyks-Markram)
        self.u = np.zeros(N_syn, dtype=np.float32)
        self.x = np.ones(N_syn, dtype=np.float32)

    def reset(self):
        self.u.fill(0.0)
        self.x.fill(1.0)
