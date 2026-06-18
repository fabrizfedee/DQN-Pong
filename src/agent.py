"""
DQN agent: online network + target network + epsilon-greedy + Huber loss.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    BATCH_SIZE, BUFFER_SIZE, DEVICE, EPS_DECAY_STEPS, EPS_END, EPS_START,
    GAMMA, LR,
)
from .model import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_shape, action_size: int):
        self.action_size = action_size
        self.epsilon = EPS_START
        self.memory = ReplayBuffer(BUFFER_SIZE, state_shape)
        self.model = DQN(action_size).to(DEVICE)
        self.target_model = DQN(action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.update_target_model()

    def update_target_model(self):
        """Hard copy of the online weights into the target network."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state), device=DEVICE).unsqueeze(0)
            return self.model(x).argmax(1).item()

    def replay(self):
        """One gradient step on a minibatch from the replay buffer."""
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.as_tensor(states, device=DEVICE)
        next_states = torch.as_tensor(next_states, device=DEVICE)
        actions = torch.as_tensor(actions, device=DEVICE).long()
        rewards = torch.as_tensor(rewards, device=DEVICE)
        dones = torch.as_tensor(dones, device=DEVICE)

        q_sa = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_model(next_states).max(1).values
            target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, target)   # Huber loss (paper)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_epsilon(self, step: int):
        """Linear annealing from EPS_START to EPS_END over EPS_DECAY_STEPS."""
        f = min(step / EPS_DECAY_STEPS, 1.0)
        self.epsilon = EPS_START + f * (EPS_END - EPS_START)

    def mean_q(self, states: torch.Tensor) -> float:
        """Average max-Q on a fixed set of held-out states (Nature metric)."""
        with torch.no_grad():
            return self.model(states).max(1).values.mean().item()
