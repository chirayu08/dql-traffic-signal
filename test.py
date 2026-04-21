import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import DQN
from memory import Memory
from simulation import Simulation
from generator import TrafficGenerator
import numpy as np

NUM_STATES  = 80
NUM_ACTIONS = 4
MAX_STEPS   = 5400
N_CARS      = 1000
GREEN_DUR   = 10
YELLOW_DUR  = 4
TEST_EPS    = 10

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError("SUMO_HOME not set.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

SUMO_CMD = [
    "sumo", "-c", "intersection/sumo_config.sumocfg",
    "--no-step-log", "true", "--waiting-time-memory", "10000"
]

model = DQN(NUM_STATES, NUM_ACTIONS, lr=0.001)
model.load("models/dqn_final.pth")
model.eval()

generator = TrafficGenerator(MAX_STEPS, N_CARS)
dummy_mem = Memory(1)
sim = Simulation(model, dummy_mem, generator, SUMO_CMD,
                 MAX_STEPS, GREEN_DUR, YELLOW_DUR,
                 NUM_STATES, NUM_ACTIONS, training=False)

results = []
for ep in range(TEST_EPS):
    total_wait, _ = sim.run(episode=ep + 1000, epsilon=0.0)
    results.append(total_wait)
    print(f"Test {ep+1:2d}: {total_wait:.0f}s")

print(f"\nMean: {np.mean(results):.0f}s")
print(f"Std:  {np.std(results):.0f}s")