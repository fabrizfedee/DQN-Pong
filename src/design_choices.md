# Design choices

This document explains the implementation choices made in this repository and
how they relate to the original DQN paper (Mnih et al., *Nature* 2015).

## 1. Network architecture (`src/model.py`)

The convolutional trunk follows the paper exactly:

| Layer | Kernel | Stride | Output channels |
| ----- | ------ | ------ | --------------- |
| Conv1 | 8×8    | 4      | 32              |
| Conv2 | 4×4    | 2      | 64              |
| Conv3 | 3×3    | 1      | 64              |

followed by a fully-connected head `Linear(3136 → 512) → ReLU → Linear(512 → n_actions)`.

The input is a `(4, 84, 84)` `uint8` tensor (last 4 frames stacked). Normalization
by `255.0` is performed **inside** `forward`, so the replay buffer can store raw
`uint8` data and save ~4× memory.

## 2. Preprocessing (`src/env.py`)

Standard Atari preprocessing pipeline via Gymnasium wrappers:

- `frame_skip=4` with max-pool on the last two frames (handled by
  `AtariPreprocessing`),
- conversion to grayscale and resize to 84×84,
- a stack of the last 4 frames as observation,
- up to 30 random no-ops at episode start.

`terminal_on_life_loss=False` is used because Pong is a single-life game (the
flag matters for games like Breakout).

## 3. Replay buffer (`src/replay_buffer.py`)

A naive `deque[Transition]` is convenient but slow: every `sample()` call has to
materialize a batch of Python objects. We preallocate NumPy arrays once and
index them directly. This gives a ~10× speedup on `sample()` and keeps states
in `uint8`, fitting 100 000 transitions in ~2.7 GB of RAM.

## 4. Agent (`src/agent.py`)

- **Two networks**: `model` (online) and `target_model` (frozen target). The
  target is hard-copied every `TARGET_UPDATE_FREQ = 1000` gradient steps.
- **Optimizer**: Adam with `lr=1e-4`. The paper uses RMSProp; Adam is the modern
  default and works just as well on Pong.
- **Loss**: Huber (`smooth_l1_loss`), exactly as the paper, to limit the impact
  of large TD errors.
- **Exploration**: linear ε-decay from 1.0 to 0.02 over 150 000 steps.
- **Reward clipping**: `sign(reward)` so updates are stable across Atari games
  with very different reward scales.

## 5. Held-out Q metric (Nature Figure 2b)

The first 500 states encountered during training are stored in a fixed tensor.
At every epoch (15 000 env steps) we compute `mean(max_a Q(s, a))` over this set
and log it. Unlike episodic reward, this metric is a smooth, low-variance proxy
for value learning and historically tracks performance closely.

## 6. Differences from the original paper

| What                | Paper      | This repo    | Why                                |
| ------------------- | ---------- | ------------ | ---------------------------------- |
| Total env. steps    | 50 M       | 1.5 M        | Single-game Colab budget           |
| Replay buffer size  | 1 M        | 100 k        | Memory on Colab T4                 |
| Optimizer           | RMSProp    | Adam         | Modern default, equivalent on Pong |
| Frames between eps  | 1 M anneal | 150 k anneal | Faster exploration in fewer steps  |
| Early stopping      | none       | mean ≥ 18    | Save compute when Pong is solved   |

These choices preserve the algorithm and only adjust the *training budget* so
the project fits inside a single Colab session.

## 7. Observed training behavior

The reported run reached a final average reward of about **+4** after the
full 1.5M-step budget (100 epochs), starting from a random-play baseline of
about **−20**. The learning curves were still rising sharply at the budget
cutoff, indicating the agent had not yet saturated — consistent with the
much larger budget used in the original paper. The held-out average max-Q
grew smoothly and monotonically throughout training, the same qualitative
behavior reported in Mnih et al. (Figure 2b) and a sanity check that value
learning was stable rather than diverging.

## 8. What I would do next

- **More compute**: extend the training budget toward the paper's 50M steps
  (or, equivalently, a larger replay buffer) to confirm convergence to the
  +18/+21 score reported in the literature.
- **Double DQN** (Van Hasselt et al., 2016) to reduce the optimistic bias in
  the bootstrap target.
- **Dueling architecture** (Wang et al., 2016) for better state-value learning.
- **Prioritized Experience Replay** (Schaul et al., 2016) for sample efficiency.
- Run the agent on a second game (e.g., Breakout) to verify generality.
