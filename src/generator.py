import numpy as np
import os

class TrafficGenerator:
    def __init__(self, max_steps, n_cars):
        self.max_steps = max_steps
        self.n_cars = n_cars
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def generate_routefile(self, seed):
        np.random.seed(seed)
        timings = np.sort(np.random.randint(0, self.max_steps, self.n_cars))
        # Valid routes based on network topology
        routes = [
            "W2TL TL2E",
            "W2TL TL2N",
            "N2TL TL2S",
            "N2TL TL2E",
            "E2TL TL2W",
            "E2TL TL2S",
            "S2TL TL2N",
            "S2TL TL2W",
        ]
        route_file = os.path.join(self.project_root, "intersection", "episode_routes.rou.xml")
        with open(route_file, "w") as f:
            f.write('<routes>\n')
            f.write('  <vType id="car" accel="1.0" decel="4.5" '
                    'length="5.0" minGap="2.5" maxSpeed="13.89"/>\n')
            for i, t in enumerate(timings):
                route = routes[np.random.randint(len(routes))]
                f.write(f'  <vehicle id="v{i}" type="car" '
                        f'depart="{t}" departLane="best" departSpeed="0">\n')
                f.write(f'    <route edges="{route}"/>\n')
                f.write('  </vehicle>\n')
            f.write('</routes>\n')