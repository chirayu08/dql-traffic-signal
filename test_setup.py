import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")
try:
    from model import DQN
    print("✓ DQN imported")
except Exception as e:
    print(f"✗ DQN import failed: {e}")

try:
    from memory import Memory
    print("✓ Memory imported")
except Exception as e:
    print(f"✗ Memory import failed: {e}")

try:
    from generator import TrafficGenerator
    print("✓ TrafficGenerator imported")
except Exception as e:
    print(f"✗ TrafficGenerator import failed: {e}")

try:
    from simulation import Simulation
    print("✓ Simulation imported")
except Exception as e:
    print(f"✗ Simulation import failed: {e}")

try:
    from utils import save_plot
    print("✓ Utils imported")
except Exception as e:
    print(f"✗ Utils import failed: {e}")

print("\nTesting file paths...")
generator = TrafficGenerator(5400, 1000)
print(f"Project root: {generator.project_root}")
print(f"Routes file path: {os.path.join(generator.project_root, 'intersection', 'episode_routes.rou.xml')}")

print("\nGenerating test routes...")
generator.generate_routefile(seed=1)
route_file = os.path.join(generator.project_root, 'intersection', 'episode_routes.rou.xml')
print(f"Routes file exists: {os.path.exists(route_file)}")

if os.path.exists(route_file):
    with open(route_file, 'r') as f:
        lines = f.readlines()
        print(f"Routes file has {len(lines)} lines")
        print("First 10 lines:")
        for i, line in enumerate(lines[:10]):
            print(f"  {i}: {line.rstrip()}")

print("\nDone!")
