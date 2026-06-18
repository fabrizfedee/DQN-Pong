"""
Training loop for DQN on Atari Pong.

Run directly as a script:
    python -m src.train

or import ``train()`` from a notebook.
"""

import os
import random
from collections import deque

import numpy as np
import torch

from .agent import DQNAgent
from .config import (
    DEVICE, N_EVAL_STATES, SAVE_PATH, SEED, STEPS_PER_EPOCH, TARGET_REWARD,
    TARGET_UPDATE_FREQ, TOTAL_STEPS, TRAIN_FREQ, WARMUP_STEPS,
)
from .env import make_env


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def train(save_path: str = SAVE_PATH):
    """Train a DQN agent on Pong. Returns (agent, episode_returns, history)."""
    set_seed(SEED)
    print("Device:", DEVICE)

    env = make_env()
    env.action_space.seed(SEED)
    state, _ = env.reset(seed=SEED)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    returns: list[float] = []
    last100: deque[float] = deque(maxlen=100)
    eval_states: list[np.ndarray] = []
    eval_tensor: torch.Tensor | None = None
    history = {"epoch": [], "step": [], "reward100": [], "q_mean": [], "epsilon": []}

    episode_return, episode = 0.0, 0

    for step in range(1, TOTAL_STEPS + 1):
        agent.set_epsilon(step)

        # Collect a fixed set of held-out states for the average max-Q metric.
        if len(eval_states) < N_EVAL_STATES:
            eval_states.append(np.asarray(state, dtype=np.uint8).copy())

        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Reward clipping (sign) as in the Nature paper.
        agent.remember(state, action, float(np.sign(reward)), next_state, float(done))
        state = next_state
        episode_return += reward

        if step > WARMUP_STEPS and step % TRAIN_FREQ == 0:
            agent.replay()
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        if done:
            episode += 1
            returns.append(episode_return)
            last100.append(episode_return)
            print(
                f"episode {episode:4d} | step {step:8d} | "
                f"reward {episode_return:+5.0f} | eps {agent.epsilon:.3f}"
            )
            if len(last100) == 100 and np.mean(last100) >= TARGET_REWARD:
                print(f"early stopping: mean reward over last 100 ep = {np.mean(last100):.2f}")
                break
            state, episode_return = env.reset()[0], 0.0

        if step % STEPS_PER_EPOCH == 0:
            if eval_tensor is None:
                eval_tensor = torch.as_tensor(np.stack(eval_states), device=DEVICE)
            avg_r = float(np.mean(last100)) if last100 else 0.0
            avg_q = agent.mean_q(eval_tensor)
            epoch = step // STEPS_PER_EPOCH
            history["epoch"].append(epoch)
            history["step"].append(step)
            history["reward100"].append(avg_r)
            history["q_mean"].append(avg_q)
            history["epsilon"].append(agent.epsilon)
            print(
                f"=== epoch {epoch:3d} | reward100 {avg_r:+6.2f} | "
                f"mean Q {avg_q:6.2f} | eps {agent.epsilon:.3f} ==="
            )

    env.close()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(agent.model.state_dict(), save_path)
    print("final weights saved to:", save_path)
    return agent, returns, history


if __name__ == "__main__":
    train()
