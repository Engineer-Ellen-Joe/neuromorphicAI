import numpy as np

class Topology:
    def __init__(self, N_pre, N_post, conn_prob=0.1):
        self.N_pre = N_pre
        self.N_post = N_post
        self.conn_prob = conn_prob

        # CSR (Compressed Sparse Row) 구조
        self.conn_idx = []
        self.conn_offset = [0]

    def generate_random(self):
        for i in range(self.N_post):
            connections = np.where(np.random.rand(self.N_pre) < self.conn_prob)[0]
            self.conn_idx.extend(connections.tolist())
            self.conn_offset.append(len(self.conn_idx))

        return np.array(self.conn_idx, dtype=np.int32), np.array(self.conn_offset, dtype=np.int32)
