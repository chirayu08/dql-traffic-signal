import traci
import numpy as np

PHASE_NS_GREEN  = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN  = 2
PHASE_EW_YELLOW = 3

class Simulation:
    def __init__(self, model, memory, generator, sumo_cmd,
                 max_steps, green_duration, yellow_duration,
                 num_states, num_actions, training=True):
        self._model = model
        self._memory = memory
        self._generator = generator
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_dur = green_duration
        self._yellow_dur = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._training = training

    def run(self, episode, epsilon=0.0):
        self._generator.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)

        step = 0
        old_wait = 0
        old_state = None
        old_action = None
        total_wait = 0
        sum_neg_reward = 0

        while step < self._max_steps:
            current_state = self._get_state()
            current_wait = self._collect_waiting_times()
            reward = old_wait - current_wait

            if self._training and old_state is not None:
                self._memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)

            if step != 0 and old_action != action:
                self._set_phase(PHASE_NS_YELLOW if old_action == PHASE_NS_GREEN else PHASE_EW_YELLOW)
                self._simulate(self._yellow_dur)
                step += self._yellow_dur

            self._set_phase(action)
            self._simulate(self._green_dur)
            step += self._green_dur

            old_state = current_state
            old_action = action
            old_wait = current_wait
            total_wait += current_wait
            if reward < 0:
                sum_neg_reward += reward

        traci.close()
        return total_wait, sum_neg_reward

    def _simulate(self, steps):
        for _ in range(steps):
            traci.simulationStep()

    def _get_state(self):
        state = np.zeros(self._num_states)
        for car_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id  = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

            if   lane_pos < 7:   cell = 0
            elif lane_pos < 14:  cell = 1
            elif lane_pos < 21:  cell = 2
            elif lane_pos < 28:  cell = 3
            elif lane_pos < 40:  cell = 4
            elif lane_pos < 60:  cell = 5
            elif lane_pos < 100: cell = 6
            elif lane_pos < 160: cell = 7
            elif lane_pos < 400: cell = 8
            else:                cell = 9

            lane_map = {
                "W2TL_0": 0, "W2TL_1": 1,
                "N2TL_0": 2, "N2TL_1": 3,
                "E2TL_0": 4, "E2TL_1": 5,
                "S2TL_0": 6, "S2TL_1": 7,
            }
            group = lane_map.get(lane_id, -1)
            if group >= 0 and lane_pos < 750 and traci.vehicle.getSpeed(car_id) < 0.1:
                state[group * 10 + cell] = 1
        return state

    def _collect_waiting_times(self):
        return sum(traci.vehicle.getAccumulatedWaitingTime(v)
                   for v in traci.vehicle.getIDList())

    def _choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self._num_actions)
        return int(np.argmax(self._model.predict_one(state)))

    def _set_phase(self, phase_code):
        traci.trafficlight.setPhase("TL", phase_code)