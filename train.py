import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import DQN
from memory import Memory
from simulation import Simulation
from generator import TrafficGenerator
from utils import save_plot

# ── Hyperparameters ──────────────────────────────────────
EPISODES       = 100
MAX_STEPS      = 5400
N_CARS         = 1000
NUM_STATES     = 80
NUM_ACTIONS    = 4
GREEN_DUR      = 10
YELLOW_DUR     = 4
GAMMA          = 0.75
LR             = 0.001
MEMORY_SIZE    = 50000
BATCH_SIZE     = 100
EPSILON_START  = 1.0
EPSILON_END    = 0.01
EPSILON_DECAY  = 400

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("SUMO_HOME not set. Add it to your environment variables.")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

SUMO_CMD = [
    "sumo",
    "-c", os.path.join(os.path.dirname(__file__), "intersection/sumo_config.sumocfg"),
    "--no-step-log", "true",
    "--waiting-time-memory", "10000"
]

# ── Init ─────────────────────────────────────────────────
model     = DQN(NUM_STATES, NUM_ACTIONS, LR)
memory    = Memory(MEMORY_SIZE)
generator = TrafficGenerator(MAX_STEPS, N_CARS)
sim       = Simulation(model, memory, generator, SUMO_CMD,
                       MAX_STEPS, GREEN_DUR, YELLOW_DUR,
                       NUM_STATES, NUM_ACTIONS, training=True)

total_waits, rewards_ep, losses = [], [], []

# ── Training loop ─────────────────────────────────────────
for ep in range(1, EPISODES + 1):
    epsilon = max(EPSILON_END,
                  EPSILON_START - (EPSILON_START - EPSILON_END) * ep / EPSILON_DECAY)

    total_wait, neg_reward = sim.run(ep, epsilon)
    total_waits.append(total_wait)
    rewards_ep.append(neg_reward)

    # Train on replay memory
    if len(memory) > BATCH_SIZE:
        batch      = memory.get_samples(BATCH_SIZE)
        states     = [s[0] for s in batch]
        actions    = [s[1] for s in batch]
        rewards    = [s[2] for s in batch]
        next_states= [s[3] for s in batch]

        q_now  = model.predict_batch(states)
        q_next = model.predict_batch(next_states)

        targets = q_now.copy()
        for i in range(len(batch)):
            targets[i][actions[i]] = rewards[i] + GAMMA * max(q_next[i])

        loss = model.train_batch(states, targets)
        losses.append(loss)

    print(f"Ep {ep:3d}/{EPISODES} | wait={total_wait:8.0f}s | "
          f"reward={neg_reward:8.0f} | eps={epsilon:.3f}")

# ── Save ──────────────────────────────────────────────────
model.save("models/dqn_final.pth")
save_plot(total_waits, "waiting_time", "Total waiting time (s)", "Cumulative wait per episode")
save_plot(rewards_ep,  "rewards",      "Cumulative negative reward", "Reward per episode")
save_plot(losses,      "loss",         "MSE Loss", "Training loss", rolling=False)
print("\nDone. Model saved to models/dqn_final.pth")