"""
Fixed-size experience replay buffer.

Implemented as preallocated NumPy arrays (not a deque of tuples): this is
~10x faster to sample from and keeps frames in uint8 for memory efficiency.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape):
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = np.asarray(state, dtype=np.uint8)
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.uint8)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        i = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[i], self.actions[i], self.rewards[i],
            self.next_states[i], self.dones[i],
        )

    def __len__(self):
        return self.size
