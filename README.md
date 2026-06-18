# Deep Q-Network on Atari Pong

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Atari-0081A5)](https://gymnasium.farama.org/)
[![Colab](https://img.shields.io/badge/Run%20on-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A clean, modular PyTorch reproduction of the **Deep Q-Network** algorithm from
Mnih et al., *Human-level control through deep reinforcement learning* (**Nature, 2015**),
applied to the Atari game **Pong** via the Gymnasium / ALE interface.

The agent learns from raw 84×84 grayscale pixels and reaches a near-perfect
score (≥ +18 average over 100 episodes) in roughly **1.5M environment steps** on
a single Colab GPU.

<p align="center">
  <img src="assets/pong_agent.gif" alt="Greedy agent playing Pong" width="320">
</p>

---

## Highlights

- **Faithful to the Nature paper**: 3-layer CNN, frame stacking of 4, frame skip
  of 4, Huber loss, target network with hard updates, reward clipping, linear
  ε-greedy decay.
- **Efficient replay buffer**: preallocated NumPy arrays in `uint8`
  (~4× less memory and ~10× faster sampling than a `deque` of tuples).
- **Held-out Q-value tracking**: 500 fixed states are captured early in training
  and the average max-Q is logged every epoch (Figure 2b in the paper).
- **Clear separation of concerns**: small modules in `src/`, two reproducible
  notebooks for training and evaluation, configuration centralized in `config.py`.

---

## Results

Training was stopped via the *early-stopping* criterion (mean reward over the
last 100 episodes ≥ 18.0). On a Colab T4 GPU:

| Metric                                    | Value          |
| ----------------------------------------- | -------------- |
| Final mean reward (last 100 episodes)     | **≈ +19 / 21** |
| Environment steps to convergence          | ~1.3–1.5 M     |
| Wall-clock time (Colab T4)                | 3–5 h          |
| Replay buffer size                        | 100 000        |
| Final ε                                   | 0.02           |

<p align="center">
  <img src="assets/pong_average_reward.png" alt="Average reward per episode" width="48%">
  <img src="assets/pong_average_max_q.png"  alt="Average max-Q on held-out states" width="48%">
</p>

The reward curve climbs from −21 (random play) to +18/+19, while the average
max-Q on the held-out set grows smoothly without diverging — the same
qualitative behavior reported in the Nature paper.

---

## Repository structure

```
DQN-Pong/
├── src/
│   ├── config.py          # all hyperparameters (Nature 2015)
│   ├── env.py             # Atari Pong + preprocessing + frame stacking
│   ├── model.py           # 3-layer CNN Q-network
│   ├── replay_buffer.py   # uint8 NumPy-backed replay buffer
│   ├── agent.py           # DQNAgent: act / remember / replay / target update
│   └── train.py           # training loop + early stopping + logging
├── notebooks/
│   ├── 01_train_dqn_pong.ipynb         # end-to-end training (Colab)
│   └── 02_evaluate_and_visualize.ipynb # GIF + Nature-style plots
├── assets/                # GIF and figures shown in this README
├── report/
│   └── design_choices.md  # implementation notes vs the paper
├── requirements.txt
└── README.md
```

---

## Quickstart

### Option A — Google Colab (recommended)

1. Open `notebooks/01_train_dqn_pong.ipynb` in Colab.
2. Set the runtime to **GPU** (T4 is enough).
3. Run all cells. The trained weights are saved to `dqn_pong_final.pt` and the
   training history to `artifacts/history.json`.
4. Open `notebooks/02_evaluate_and_visualize.ipynb` to generate the GIF and the
   two Nature-style plots.

### Option B — Local

```bash
git clone https://github.com/<your-username>/DQN-Pong.git
cd DQN-Pong
pip install -r requirements.txt
AutoROM --accept-license     # one-time, downloads the Atari ROMs
python -m src.train
```

A CUDA-capable GPU is strongly recommended.

---

## Key hyperparameters

All defined in `src/config.py`:

| Group       | Parameter             | Value     |
| ----------- | --------------------- | --------- |
| Schedule    | `TOTAL_STEPS`         | 1 500 000 |
|             | `WARMUP_STEPS`        | 10 000    |
|             | `STEPS_PER_EPOCH`     | 15 000    |
| Replay      | `BUFFER_SIZE`         | 100 000   |
|             | `BATCH_SIZE`          | 32        |
| Optimizer   | `LR` (Adam)           | 1e-4      |
|             | `GAMMA`               | 0.99      |
| DQN         | `TRAIN_FREQ`          | 4         |
|             | `TARGET_UPDATE_FREQ`  | 1 000     |
| ε-greedy    | `EPS_START → EPS_END` | 1.0 → 0.02|
|             | `EPS_DECAY_STEPS`     | 150 000   |
| Stopping    | `TARGET_REWARD`       | 18.0      |

Differences from the original Nature paper (compute-budget driven) are
documented in [`report/design_choices.md`](report/design_choices.md).

---

## How DQN works (in 30 seconds)

DQN learns a parametric approximation $Q_\theta(s, a)$ of the optimal
action-value function by minimizing the Bellman residual on minibatches of
*past* transitions drawn from an experience replay buffer:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}
\left[\,\big(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\big)^2\,\right]
$$

Two well-known tricks stabilize training:
1. **Experience replay** breaks the temporal correlation between consecutive
   transitions, making the gradient estimator closer to i.i.d.
2. **Target network** $Q_{\theta^-}$ — a periodically frozen copy of the online
   network — prevents the bootstrap target from chasing a moving estimate.

See [`report/design_choices.md`](report/design_choices.md) for a more detailed
walkthrough of every component.

---

## References

The original paper:

```bibtex
@article{mnih2015human,
  title   = {Human-level control through deep reinforcement learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu,
             Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex
             and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski,
             Georg and others},
  journal = {Nature},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  year    = {2015},
  publisher = {Nature Publishing Group}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).
