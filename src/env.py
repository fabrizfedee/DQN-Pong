"""
Atari Pong environment factory.

Reproduces the standard DQN preprocessing pipeline:
  - frame skip = 4 (action repeat with max-pool on the last two frames)
  - grayscale conversion
  - resize to 84x84
  - stack of the last 4 frames as observation (shape: 4 x 84 x 84, uint8)
"""

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

try:
    gym.register_envs(ale_py)
except Exception:
    pass


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create a preprocessed, frame-stacked Pong environment.

    Args:
        render_mode: pass ``"rgb_array"`` to enable ``env.render()`` for GIFs.
    """
    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env
