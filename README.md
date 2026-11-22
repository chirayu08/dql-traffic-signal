Deep Q-Learning Agent for Adaptive Traffic Signal Control Using SUMO

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.26.0-green.svg)](https://eclipse.dev/sumo/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/Conference-ASIT--2026-success.svg)](https://icasit.thehinweis.com/2026/)

A Deep Q-Network (DQN) based reinforcement learning system for **adaptive traffic signal control** at a single 4-way intersection using the **SUMO microscopic traffic simulator** and the **TraCI Python API**.

The system learns signal-control policies from simulated traffic conditions and dynamically selects traffic-light phases with the objective of reducing cumulative vehicle waiting time.

> **Research result:** After 400 training episodes, the proposed DQN agent achieved a **76.2% reduction in cumulative waiting time compared with the fixed-time baseline** and a **44.7% reduction compared with the gap-based actuated controller**, evaluated over 10 held-out traffic seeds.

---

## Research Paper

The research paper associated with this project was presented at the **Hinweis Fifth International Conference on Advances in Software Engineering and Information Technology (ASIT-2026)**.

**Paper Title:**

Deep Q-Learning Agent for Adaptive Traffic Signal Control Using SUMO Microscopic Simulation

**Conference Details:**

- Conference: ASIT-2026
- Conference Organizer: Hinweis Research
- Presentation Date: September 27, 2026
- Institution: VIT-AP University, Amaravathi, Andhra Pradesh, India
- Research Area: Deep Reinforcement Learning and Intelligent Traffic Signal Control

### Published Research Paper

**[Read the Full Research Paper (PDF)](./paper/Deep_Q_Learning_Adaptive_Traffic_Signal_Control.pdf)**

### Certificate of Presentation & Publication

**[View ASIT-2026 Conference Certificate](./docs/assets/ASIT_2026_Certificate.jpeg)**

The paper presents the target-network DQN methodology, simulation environment, state/action/reward formulation, experimental configuration, baseline comparisons, training analysis, limitations, and future research directions.

---

## Overview

Traditional fixed-time traffic signal controllers operate according to predefined cycles and do not respond effectively to changing traffic demand. This project applies **model-free reinforcement learning** to traffic signal control.

The DQN agent interacts with a SUMO simulation environment through TraCI:

```text
Traffic Simulation (SUMO)
         |
         v
    Traffic State
  80-dimensional vector
         |
         v
      DQN Agent
         |
         v
   Signal Phase Action
         |
         v
    SUMO Traffic Light
         |
         v
     Reward Signal
         |
         v
 Experience Replay Buffer
         |
         v
   Update Online Network
         |
         v
 Synchronize Target Network
         |
         └──────────────> Agent learns
```

The paper uses an 80-dimensional binary representation of stopped-vehicle positions across eight incoming lanes and four discrete traffic-signal actions.

---

## Objectives

- Develop an adaptive traffic signal controller using Deep Q-Learning.
- Reduce cumulative vehicle waiting time.
- Enable the controller to respond to stochastic traffic demand.
- Train and evaluate the agent safely in a microscopic traffic simulation.
- Improve training stability using experience replay and a periodically synchronized target network.
- Compare the proposed controller against fixed-time and gap-based actuated traffic controllers.
- Provide a reproducible research implementation for further experimentation.

---

## Methodology

### 1. Traffic Environment

The environment is implemented using **SUMO (Simulation of Urban MObility)**.

The simulated network contains:

- One 4-way intersection.
- Four incoming approaches.
- Two lanes per incoming approach.
- Four traffic signal phases.
- Stochastically generated vehicle traffic.
- 90-minute simulation episodes.
- 1,000 vehicles per training episode.

SUMO and TraCI provide bidirectional communication between the reinforcement learning agent and the traffic simulation.

### 2. State Representation

The agent receives an **80-dimensional binary state vector**.

Each of the eight incoming lanes is divided into ten road cells. A cell is assigned:

- `1` → A stationary vehicle is present.
- `0` → No stationary vehicle is present.

A vehicle is considered stationary when its speed is below the defined threshold.

This allows the agent to identify queue formation and congestion near the intersection.

### 3. Action Space

The agent chooses between four discrete signal phases:

| Action | Phase | Description |
|---|---|---|
| `0` | NS Green | North-South traffic receives green |
| `1` | NS Yellow | North-South clearing phase |
| `2` | EW Green | East-West traffic receives green |
| `3` | EW Yellow | East-West clearing phase |

A 4-second yellow transition is inserted when switching phases, followed by a 10-second decision interval.

### 4. Reward Function

The reward is based on the change in cumulative vehicle waiting time:

```text
rₜ = Wₜ₋₁ − Wₜ
```

Where `Wₜ` represents the cumulative waiting time of active vehicles.

Therefore:

- Waiting time decreases → Positive reward.
- Waiting time increases → Negative reward.

This directly aligns the learning objective with traffic-delay reduction.

---

## DQN Architecture

The DQN maps the 80-dimensional traffic state to four Q-values.

```text
Input
 80 neurons
    |
    v
Dense Layer
 400 neurons
    |
   ReLU
    |
    v
Dense Layer
 400 neurons
    |
   ReLU
    |
    v
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

### Target Network

Unlike the earlier implementation, the updated DQN includes a separate target network.

The target network is a periodically synchronized copy of the online Q-network.

- Online network: Learns and updates its parameters.
- Target network: Provides more stable Bellman target values.
- Synchronization: Every 5 training episodes.

The target network reduces the moving-target problem during Q-learning and is associated with substantially lower loss-spike instability in the reported training run.

### Experience Replay

A circular replay buffer stores transitions:

`(state, action, reward, next_state)`

After collecting sufficient experience, random mini-batches of 100 transitions are sampled from the buffer for gradient updates.

This reduces temporal correlation between consecutive experiences and improves training stability.

---

## Reinforcement Learning Pipeline

```text
Initialize Online DQN
         |
         v
Initialize Target Network
         |
         v
Start SUMO Episode
         |
         v
Observe 80-D State
         |
         v
Epsilon-Greedy Action Selection
         |
         v
Apply Traffic Signal Action
         |
         v
Advance SUMO Simulation
         |
         v
Calculate Waiting-Time Reward
         |
         v
Store Transition in Replay Buffer
         |
         v
Sample Mini-Batch
         |
         v
Calculate Target Q-Values
         |
         v
Update Online DQN
         |
         v
Synchronize Target Network
         |
         v
Repeat Until Episode Ends
         |
         v
Start Next Episode
```

The agent uses epsilon-greedy exploration, with epsilon decaying linearly from `1.0` toward `0.01` over 400 episodes.

The target network is synchronized every five episodes.

---

## Experimental Configuration

| Parameter | Value |
|---|---:|
| Training episodes | 400 |
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
| Target network synchronization | Every 5 episodes |
| Hidden layers | 2 × 400 |
| Activation | ReLU |

These values correspond to the updated experimental configuration reported in the research paper.

---

## Results

The DQL controller was compared against two non-learning traffic signal controllers over ten held-out traffic seeds.

### Performance Comparison

| Method | Mean Cumulative Waiting Time | Standard Deviation |
|---|---:|---:|
| Fixed-Time Controller | 67,613 s | ±3,008 s |
| Gap-Based Actuated Controller | 29,070 s | ±691 s |
| **DQL Agent** | **16,076 s** | **±687 s** |

### Performance Improvements

| Comparison | Waiting-Time Reduction |
|---|---:|
| DQL vs Fixed-Time | **76.2%** |
| DQL vs Actuated | **44.7%** |
| Actuated vs Fixed-Time | 57.0% |

The proposed DQL agent achieved the lowest mean cumulative waiting time among the three controllers.

The comparison against the actuated controller is particularly important because it demonstrates an improvement over a demand-responsive, non-learning traffic control approach.

### Training Behaviour

The updated paper analyses three major training indicators:

1. **Cumulative Waiting Time:** The 10-episode moving average decreases from approximately 85,000–100,000 seconds during early exploration to approximately 15,000–18,000 seconds by the end of training.

2. **Cumulative Reward:** The 10-episode moving average improves from approximately -10,000 to approximately -2,100 by episode 400.

3. **Training Loss:** The MSE loss decreases from an early range of approximately 1,500–3,500 to below 200 by episode 300, with progressively smaller fluctuations.

The results indicate smoother training behaviour with the target network enabled.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.11** | Main implementation language |
| **PyTorch 2.11** | DQN implementation and training |
| **SUMO 1.26.0** | Microscopic traffic simulation |
| **TraCI** | Python interface for controlling SUMO |
| **NumPy 1.24** | Numerical computation |
| **Matplotlib 3.7** | Training and result visualization |

---

## Repository Structure

```text
dql-traffic-signal/
│
├── README.md
├── LICENSE
│
├── paper/
│   └── Deep_Q_Learning_Adaptive_Traffic_Signal_Control.pdf
│
├── docs/
│   └── assets/
│       └── ASIT_2026_Certificate.jpg
│
├── intersection/
│
├── src/
│
├── plots/
│
├── train.py
├── test.py
├── test_setup.py
│
└── requirements.txt
```

The folder structure above represents the intended organization. Refer to the actual repository files for the current implementation.

---

## Installation

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
pip install torch numpy matplotlib
```

SUMO must be installed separately, and the `SUMO_HOME` environment variable must be configured so that the TraCI Python package is accessible.

If a `requirements.txt` file is available, dependencies can also be installed using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

After installing SUMO and the Python dependencies:

### 1. Test the Setup

```bash
python test_setup.py
```

### 2. Train the Agent

```bash
python train.py
```

The training process:

1. Generates traffic routes.
2. Starts the SUMO simulation.
3. Collects the current traffic state.
4. Selects a signal phase using the DQN policy.
5. Executes the action through TraCI.
6. Calculates the waiting-time reward.
7. Stores the transition in replay memory.
8. Updates the online neural network.
9. Synchronizes the target network periodically.
10. Records training metrics.

### 3. Evaluate the Agent

```bash
python test.py
```

Use the trained model and evaluation configuration expected by the repository.

---

## Evaluation

The primary performance metric is **cumulative vehicle waiting time**.

Additional metrics analysed in the paper include:

- Cumulative reward.
- DQN training loss.
- Moving average waiting time.
- Standard deviation across held-out traffic seeds.

The evaluation uses ten held-out seeds (9001–9010), separate from the 400 training-episode seeds.

This helps assess how the learned policy performs under traffic conditions not used during training.

---

## Limitations

The current research implementation has several limitations:

- The experiment covers only a **single isolated intersection**.
- The state representation does not explicitly encode the current signal phase or elapsed phase time.
- Evaluation is conducted entirely within SUMO simulation.
- Results are reported from a single training run.
- Real-world deployment requires additional sensor validation, robustness testing, and fail-safe mechanisms.

The standard deviations reported across ten evaluation seeds represent test-time traffic variability, not variation across independently trained models.

---

## Future Work

Planned extensions include:

1. **Double DQN** to reduce Q-value overestimation.
2. **Dueling DQN** for improved state-value and action-advantage estimation.
3. Enriching the state representation with current signal phase and elapsed phase time.
4. **Multi-intersection coordination** using cooperative reinforcement learning.
5. Repeating training across multiple random initializations to evaluate training variability.
6. Transfer learning from simulated environments to real-world traffic sensor data.
7. Developing fail-safe mechanisms for eventual real-world deployment.

---

## References

The complete bibliography is available in the accompanying research paper.

Key references include:

- Sutton and Barto — *Reinforcement Learning: An Introduction*.
- Mnih et al. — *Human-level control through deep reinforcement learning*.
- Behrisch et al. — *SUMO: Simulation of Urban MObility*.
- Vidali et al. — *A Deep Reinforcement Learning Approach to Adaptive Traffic Signals Management*.
- Genders and Razavi — *Using a Deep Reinforcement Learning Agent for Traffic Signal Control*.
- Wei et al. — *CoLight: Learning Network-Level Cooperation for Traffic Signal Control*.

---

## Citation

If you use this implementation or research in your work, please cite the following paper:

**Deep Q-Learning Agent for Adaptive Traffic Signal Control Using SUMO Microscopic Simulation.**

Presented at the Hinweis Fifth International Conference on Advances in Software Engineering and Information Technology (ASIT-2026), September 27, 2026.

---

## License

This project is released under the **MIT License**.

---

## Author

**Chirayu Jaju**

School of Computer Science and Engineering  
VIT-AP University, Andhra Pradesh, India

GitHub: [@chirayu08](https://github.com/chirayu08)

Repository: [dql-traffic-signal](https://github.com/chirayu08/dql-traffic-signal)

---

## Acknowledgements

The authors thank the SUMO development team at DLR for maintaining an excellent open-source traffic simulator and the PyTorch community for the deep learning framework used in this work.

We also acknowledge the guidance and support received from our faculty during the development of this research project.
