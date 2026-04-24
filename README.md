# dql-traffic-signal

A Deep Q-Network agent that learns to control traffic lights — trained entirely inside SUMO without ever touching real hardware.

---

## What this does

Fixed-time traffic lights are dumb. They run the same cycle whether there are 3 cars or 300. This project replaces that with a DQN agent that watches stopped vehicles across 8 approach lanes and decides which phase to run next — adapting in real time to whatever traffic shows up.

After 100 training episodes (each one simulating 90 minutes of traffic with 1,000 vehicles), the agent cuts average cumulative waiting time by **34.4%** against a fixed-time baseline.

---

## Results

| Controller | Avg. Cumulative Wait | Std Dev |
|---|---|---|
| Fixed-time (31 s cycles) | ~95,000 s | ±15,000 s |
| DQL agent (ours) | ~63,000 s | ±8,200 s |

Beyond raw wait time, the standard deviation dropped nearly in half — meaning the learned policy is more consistent across different traffic patterns, not just faster on average.

Training goes through three recognisable phases:
- **Episodes 1–40**: mostly random exploration, waiting times all over the place
- **Episodes 40–65**: Q-values start forming but early estimates are noisy — brief performance dip
- **Episodes 65–100**: policy locks in, 10-ep moving average drops from ~95k to ~52k seconds

---

## How it works

**State** — 80-dimensional binary vector. Eight lanes × 10 cells each. A cell is `1` if a stationary vehicle (< 0.1 m/s) occupies it. Cells near the stop line are shorter for finer resolution where it matters.

**Actions** — Four signal phases (NS green, NS yellow, EW green, EW yellow). Switching phases automatically inserts a 4-second yellow before the new green.

**Reward** — `r_t = W_{t-1} - W_t`, the reduction in total accumulated vehicle waiting time. Congestion increasing = negative reward. Clean and dense, no proxy shaping needed.

**Network** — `80 → 400 → 400 → 4` with ReLU activations, Adam optimiser, MSE loss. No target network in this baseline (see limitations).

**Exploration** — ε-greedy, linear decay from 1.0 → 0.01 over 400 episodes. At episode 100, ε ≈ 0.75 — there's still headroom if you train longer.

---

## Stack

- Python 3.11
- PyTorch 2.1
- SUMO 1.26.0
- NumPy 1.24
- Matplotlib 3.7

---

## Getting started

**Install SUMO**

```bash
# Ubuntu
sudo apt install sumo sumo-tools sumo-doc

# macOS
brew install sumo
```

Set the environment variable:
```bash
export SUMO_HOME=/usr/share/sumo   # adjust to your install path
```

**Install Python dependencies**

```bash
pip install torch numpy matplotlib
pip install traci  # or install via sumo-tools
```

**Train the agent**

```bash
python training_main.py
```

Training runs 100 episodes by default. Results and plots are saved automatically.

**Run a single evaluation episode**

```bash
python testing_main.py
```

---

## Project layout

```
dql-traffic-signal/
├── training_main.py        # entry point for training
├── testing_main.py         # evaluate a saved model
├── generator.py            # stochastic route file generation
├── model.py                # DQN architecture
├── memory.py               # experience replay buffer
├── visualization.py        # training curve plots
└── intersection/
    ├── environment.py      # TraCI wrapper, state/reward logic
    └── sumo_files/         # .net.xml, .rou.xml, .cfg
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 100 |
| Simulation duration | 5,400 s (90 min) |
| Vehicles per episode | 1,000 |
| State size | 80 |
| Actions | 4 |
| Green phase | 10 s |
| Yellow phase | 4 s |
| Learning rate | 1e-3 |
| Discount factor γ | 0.75 |
| Replay buffer | 50,000 |
| Batch size | 100 |
| ε (start → end) | 1.0 → 0.01 |
| ε decay horizon | 400 episodes |
| Hidden layers | 2 × 400, ReLU |

---

## Limitations worth knowing

- **No target network.** This causes the loss spikes visible in training (especially around episode 58). A periodic frozen copy of the Q-network would stabilise this.
- **Single intersection.** The setup is one 4-way junction. Real networks need coordination between multiple agents.
- **Incomplete state.** The current phase and elapsed phase time aren't in the state vector. The agent can't reason about how long it's been in the current phase.
- **Simulation only.** Tested entirely in SUMO. Real-world deployment would need sensor integration and additional robustness work.
- **Still exploring at episode 100.** ε ≈ 0.75 at termination. Training to 200–400 episodes should improve results further.

---

## What's next

- [ ] Double DQN to reduce Q-value overestimation
- [ ] Dueling DQN architecture
- [ ] Add target network with periodic sync
- [ ] Include current phase + elapsed time in state
- [ ] Multi-intersection coordination with cooperative agents
- [ ] Transfer to real-world loop detector data

---

## Citation

If you use this in your work:

```bibtex
@misc{dql-traffic-signal,
  title   = {Deep Q-Learning Agent for Adaptive Traffic Signal Control Using SUMO},
  year    = {2024},
  url     = {https://github.com/chirayu08/dql-traffic-signal}
}
```

Primary reference this builds on: Vidali et al., *A deep reinforcement learning approach to adaptive traffic signals management*, AI*IA Workshop IAS-UrbMob, 2019.

---

## License

MIT
