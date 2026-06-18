"""
Hyperparameters for DQN on Atari Pong.

Values follow the Nature 2015 paper (Mnih et al., "Human-level control through
deep reinforcement learning") with minor adjustments for Colab-friendly training:
a 100k replay buffer (vs 1M in the paper) and 1.5M environment steps (vs 50M),
which is enough to reach a near-perfect score on Pong specifically.
"""

import torch

# --- Training schedule ---
TOTAL_STEPS = 1_500_000
WARMUP_STEPS = 10_000          # random actions to populate the replay buffer
STEPS_PER_EPOCH = 15_000       # logging / held-out Q evaluation cadence

# --- Replay buffer / optimization ---
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4

# --- DQN-specific frequencies ---
TRAIN_FREQ = 4                 # gradient step every K environment steps
TARGET_UPDATE_FREQ = 1_000     # hard copy online -> target every K steps

# --- Epsilon-greedy schedule ---
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_STEPS = 150_000      # linear decay over this many steps

# --- Evaluation ---
N_EVAL_STATES = 500            # held-out states for the average max-Q metric
TARGET_REWARD = 18.0           # early-stopping threshold (avg over last 100 ep.)

# --- Misc ---
SEED = 0
SAVE_PATH = "dqn_pong_final.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
