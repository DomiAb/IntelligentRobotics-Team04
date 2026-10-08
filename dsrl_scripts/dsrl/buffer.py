"""
CPU-side replay buffer for DSRL / SAC training.

Stores (s, a, r, s', done) tuples as pre-allocated numpy arrays and supports
uniform random sampling.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim: int, act_dim: int, capacity: int = int(1e6)):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.s  = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.capacity, act_dim),   dtype=np.float32)
        self.r  = np.zeros((self.capacity, 1),          dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.capacity, 1),          dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]
