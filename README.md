[README.md](https://github.com/user-attachments/files/31129932/README.md)

# Deep Q-Learning Agent for Adaptive Traffic Signal Control Using SUMO

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.26.0-green.svg)](https://eclipse.dev/sumo/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Q-Network (DQN) based reinforcement learning system for **adaptive traffic signal control** at a single 4-way intersection using the **SUMO microscopic traffic simulator** and the **TraCI Python API**.

The system learns signal-control policies from simulated traffic conditions and dynamically selects traffic-light phases with the objective of reducing cumulative vehicle waiting time.

> **Research result:** After 100 training episodes, the DQL agent achieved a **34.4% reduction in average vehicle waiting time** compared with the fixed-time baseline reported in the accompanying paper.

---

##  Research Paper

The complete research paper is included in this repository:

**[Deep Q-Learning Agent for Adaptive Traffic Signal Control Using SUMO Microscopic Simulation](./paper/Deep_Q_Learning_Adaptive_Traffic_Signal_Control.pdf)**

The paper describes the methodology, environment, state/action/reward formulation, DQN architecture, training procedure, experimental results, limitations, and future research directions. fileciteturn0file0L2-L22

---

##  Overview

Traditional fixed-time traffic signal controllers operate according to predefined cycles and do not respond effectively to changing traffic demand. This project applies **model-free reinforcement learning** to traffic signal control.

The DQN agent interacts with a SUMO simulation environment through TraCI:

```text
Traffic Simulation (SUMO)
          │
          ▼
     Traffic State
   80-dimensional vector
          │
          ▼
       DQN Agent
          │
          ▼
    Signal Phase Action
          │
          ▼
     SUMO Traffic Light
          │
          ▼
      Reward Signal
          │
          └──────────────► Agent learns
```

The paper uses an 80-dimensional binary representation of stopped-vehicle positions across eight incoming lanes and four discrete traffic-signal actions. fileciteturn0file0L165-L186

---

##  Objectives

- Develop an adaptive traffic signal controller using Deep Q-Learning.
- Reduce cumulative vehicle waiting time.
- Enable the controller to respond to stochastic traffic demand.
- Train and evaluate the agent safely in a microscopic traffic simulation.
- Provide a reproducible research implementation for further experimentation.

---

##  Methodology

### 1. Traffic Environment

The environment is implemented using **SUMO (Simulation of Urban MObility)**.

The simulated network contains:

- One 4-way intersection
- Four incoming approaches
- Two lanes per incoming approach
- Four traffic signal phases
- Stochastically generated vehicle traffic
- 90-minute simulation episodes

SUMO and TraCI provide bidirectional communication between the reinforcement learning agent and the traffic simulation. fileciteturn0file0L126-L140

### 2. State Representation

The agent receives an **80-dimensional binary state vector**.

Each of the eight incoming lanes is divided into ten road cells. A cell is assigned:

- `1` → a stationary vehicle is present
- `0` → no stationary vehicle is present

This allows the agent to identify queue formation and congestion near the intersection. fileciteturn0file0L165-L179

### 3. Action Space

The agent chooses between four discrete signal phases:

| Action | Phase | Description |
|---|---|---|
| `0` | NS Green | North-South traffic receives green |
| `1` | NS Yellow | North-South clearing phase |
| `2` | EW Green | East-West traffic receives green |
| `3` | EW Yellow | East-West clearing phase |

A 4-second yellow transition is inserted when switching phases, followed by a 10-second green phase. fileciteturn0file0L141-L151 fileciteturn0file0L180-L186

### 4. Reward Function

The reward is based on the change in cumulative vehicle waiting time:

```text
rₜ = Wₜ₋₁ − Wₜ
```

where `Wₜ` represents the cumulative waiting time of active vehicles.

Therefore:

- Waiting time decreases → positive reward
- Waiting time increases → negative reward

This directly aligns the learning objective with traffic-delay reduction. fileciteturn0file0L187-L199

---

##  DQN Architecture

The DQN maps the 80-dimensional traffic state to four Q-values.

```text
Input
  80 neurons
     │
     ▼
Dense Layer
  400 neurons
     │
    ReLU
     │
     ▼
Dense Layer
  400 neurons
     │
    ReLU
     │
     ▼
Output Layer
  4 Q-values
```

The network uses:

- **Optimizer:** Adam
- **Learning rate:** `1 × 10⁻³`
- **Loss:** Mean Squared Error (MSE)
- **Activation:** ReLU
- **Replay memory:** 50,000 transitions
- **Batch size:** 100

The reported implementation does not use a target network, which contributes to the observed training-loss instability. fileciteturn0file0L200-L227

---

##  Reinforcement Learning Pipeline

```text
Initialize DQN
      │
      ▼
Start SUMO Episode
      │
      ▼
Observe 80-D State
      │
      ▼
ε-Greedy Action Selection
      │
      ▼
Apply Traffic Signal Action
      │
      ▼
Advance SUMO Simulation
      │
      ▼
Calculate Waiting-Time Reward
      │
      ▼
Store Transition in Replay Buffer
      │
      ▼
Sample Mini-Batch
      │
      ▼
Update DQN
      │
      ▼
Repeat Until Episode Ends
      │
      ▼
Start Next Episode
```

The agent uses epsilon-greedy exploration, with epsilon decaying from `1.0` toward `0.01` over a 400-episode horizon. fileciteturn0file0L228-L240

---

##  Experimental Configuration

| Parameter | Value |
|---|---:|
| Training episodes | 100 |
| Simulation duration | 5,400 s (90 min) |
| Vehicles per episode | 1,000 |
| State size | 80 |
| Number of actions | 4 |
| Green phase | 10 s |
| Yellow phase | 4 s |
| Optimizer | Adam |
| Learning rate | `1 × 10⁻³` |
| Discount factor γ | 0.75 |
| Replay memory | 50,000 |
| Batch size | 100 |
| Initial ε | 1.0 |
| Final ε | 0.01 |
| ε decay horizon | 400 episodes |
| Hidden layers | 2 × 400 |
| Activation | ReLU |

These values correspond to the experimental configuration reported in the paper. fileciteturn0file0L241-L273

---

##  Results

The DQL controller was compared against a fixed-time traffic signal controller over ten random seeds.

| Method | Average Waiting Time | Standard Deviation | Improvement |
|---|---:|---:|---:|
| Fixed-Time Controller | ~95,000 s | ±15,000 s | Baseline |
| **DQL Agent** | **~63,000 s** | **±8,200 s** | **34.4% reduction** |

The DQL agent reduced average vehicle waiting time by **34.4%** and also showed lower variability across traffic-demand seeds. fileciteturn0file0L300-L314

### Training Behaviour

The reported training curve shows three broad phases:

1. **Episodes 1–40 — Exploration:** predominantly random actions due to high epsilon.
2. **Episodes 40–65 — Transition:** Q-values begin influencing decisions, with temporary instability.
3. **Episodes 65–100 — Improvement:** the moving average of waiting time decreases as the policy improves.

The paper reports that the 10-episode moving average decreases from approximately 95,000 s to 52,000 s during the final training phase. fileciteturn0file0L243-L278

---

##  Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.11** | Main implementation language |
| **PyTorch 2.11** | DQN implementation and training |
| **SUMO 1.26.0** | Microscopic traffic simulation |
| **TraCI** | Python interface for controlling SUMO |
| **NumPy 1.24** | Numerical computation |
| **Matplotlib 3.7** | Training and result visualization |

The software versions are specified in the paper's implementation description. fileciteturn0file0L160-L164

---

##  Suggested Repository Structure

```text
dql-traffic-signal/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── paper/
│   └── Deep_Q_Learning_Adaptive_Traffic_Signal_Control.pdf
│
├── src/
│   ├── agent.py
│   ├── generator.py
│   ├── model.py
│   ├── environment.py
│   └── train.py
│
├── sumo/
│   ├── network/
│   ├── routes/
│   └── configuration/
│
├── results/
│   ├── waiting_time.png
│   ├── reward.png
│   └── training_loss.png
│
└── requirements.txt
```

Adjust the source-file names above to match the actual files in the repository.

---

##  Installation

### Prerequisites

Install:

- Python 3.11
- SUMO 1.26.0
- Git

Verify the installations:

```bash
python --version
sumo --version
```

### Clone the Repository

```bash
git clone https://github.com/chirayu08/dql-traffic-signal.git
cd dql-traffic-signal
```

### Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet present, the paper specifies the primary software dependencies as PyTorch, NumPy, and Matplotlib, with SUMO/TraCI required for simulation. fileciteturn0file0L160-L164

---

##  Running the Project

After installing SUMO and the Python dependencies:

```bash
python train.py
```

The training process should:

1. Generate traffic routes.
2. Start the SUMO simulation.
3. Collect the current traffic state.
4. Select a signal phase using the DQN policy.
5. Execute the action through TraCI.
6. Calculate the waiting-time reward.
7. Store the transition in replay memory.
8. Train the neural network from sampled experiences.
9. Record training metrics.

Use the actual entry-point filename from the repository if it differs from `train.py`.

---

##  Evaluation

The primary performance metric is **cumulative vehicle waiting time**.

Additional metrics used in the paper include:

- Cumulative reward
- DQN training loss
- Moving average waiting time
- Standard deviation across traffic seeds

The accompanying paper analyses all three training curves and compares the learned policy with a fixed-time baseline. fileciteturn0file0L279-L314

---

##  Limitations

The current research implementation has several limitations:

- The experiment focuses on a **single isolated intersection**.
- The DQN does not use a target network.
- The state representation does not explicitly include the current signal phase or elapsed phase time.
- Evaluation is performed entirely in simulation.
- At the end of the reported 100 episodes, epsilon remains around `0.75`, meaning substantial exploration is still occurring.

These limitations are identified in the paper and motivate the proposed future improvements. fileciteturn0file0L315-L335

---

##  Future Work

Planned extensions include:

1. **Double DQN** to reduce Q-value overestimation.
2. **Dueling DQN** for improved value/action representation.
3. **Target network** with periodic parameter synchronization.
4. Enriching the state with **current phase and elapsed phase time**.
5. **Multi-intersection coordination** using cooperative DQN agents.
6. **Transfer learning** from simulated environments to real-world traffic sensor data.

These directions are outlined in the research paper. fileciteturn0file0L336-L352

---

##  References

The complete bibliography is available in the accompanying paper, including foundational works on reinforcement learning, DQN, SUMO, experience replay, adaptive traffic signals, Double DQN, and Dueling DQN. fileciteturn0file0L358-L397

Key references include:

- Sutton & Barto — *Reinforcement Learning: An Introduction*
- Mnih et al. — *Human-level control through deep reinforcement learning*
- Behrisch et al. — *SUMO — Simulation of Urban MObility*
- Vidali et al. — *A deep reinforcement learning approach to adaptive traffic signals management*
- Chu et al. — *Multi-agent deep reinforcement learning for large-scale traffic signal control*

---

##  Citation

If you use this project or the accompanying research paper in your work, please cite the paper:

```text
Deep Q-Learning Agent for Adaptive Traffic Signal Control Using
SUMO Microscopic Simulation.
```

---

##  License

This project is released under the **MIT License**, as stated in the accompanying paper. fileciteturn0file0L348-L352

---

##  Author

**Chirayu Jaju**

GitHub: [@chirayu08](https://github.com/chirayu08)

Repository: [dql-traffic-signal](https://github.com/chirayu08/dql-traffic-signal)

---

## Acknowledgements

This work acknowledges the **SUMO development team at DLR** for maintaining the SUMO traffic simulation platform and the **PyTorch community** for the deep learning framework used in the project. fileciteturn0file0L353-L357
