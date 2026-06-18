"""
Convolutional Q-network from Mnih et al., Nature 2015.

Input : (B, 4, 84, 84) uint8 frame stack
Output: (B, n_actions) Q-values
"""

import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        # 3 conv layers, same kernel/stride as the Nature paper
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # Normalize uint8 [0,255] -> float32 [0,1] inside the model so the
        # replay buffer can stay in uint8 and save ~4x memory.
        x = self.conv(x.float() / 255.0)
        return self.fc(x.reshape(x.size(0), -1))
