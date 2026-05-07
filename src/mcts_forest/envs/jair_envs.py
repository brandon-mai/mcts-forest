import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from mcts_forest.envs.procedural import (
    river_swim_step, river_swim_rollout,
    four_rooms_step, four_rooms_rollout,
    passenger_grid_step, passenger_grid_rollout,
    sysadmin_ring_step, sysadmin_ring_rollout
)

class FactoredRiverSwimEnv(gym.Env):
    def __init__(self, num_rivers=4, num_locations=8, time_limit=35, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.num_rivers = num_rivers
        self.num_locations = num_locations
        self.time_limit = time_limit
        self.action_space = spaces.Discrete(1 << num_rivers)
        self.observation_space = spaces.Discrete((num_locations ** num_rivers) * (time_limit + 1))
        self.reset()

    def get_fingerprint(self):
        return ("FactoredRiverSwim", self.num_rivers, self.num_locations, self.time_limit)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.zeros(self.num_rivers, dtype=np.int32)
        self.time = 0
        self.s = self.get_state()
        return self.s, {}

    def get_state(self) -> int:
        spatial_id = 0
        factor = 1
        for river in range(self.num_rivers):
            spatial_id += self.positions[river] * factor
            factor *= self.num_locations
        return int(spatial_id * (self.time_limit + 1) + self.time)

    def set_state(self, state_idx: int):
        self.s = int(state_idx)
        self.time = self.s % (self.time_limit + 1)
        spatial_id = self.s // (self.time_limit + 1)
        for river in range(self.num_rivers):
            self.positions[river] = spatial_id % self.num_locations
            spatial_id //= self.num_locations

    def step(self, action: int):
        action_bits = [(action >> i) & 1 for i in range(self.num_rivers)]
        reward = self._compute_reward(self.positions, action_bits)
        
        # Stochastic transitions
        for river in range(self.num_rivers):
            self.positions[river] = self._sample_next_pos(self.positions[river], action_bits[river])
            
        self.time += 1
        terminated = self.time >= self.time_limit
        self.s = self.get_state()
        return self.s, reward, terminated, False, {}

    def _sample_next_pos(self, pos, action_bit):
        goal = self.num_locations - 1
        if action_bit == 0: return max(0, pos - 1)
        p = self.np_random.random()
        if pos == 0: return 0 if p < 0.4 else 1
        if pos == goal: return goal - 1 if p < 0.4 else goal
        if p < 0.05: return pos - 1
        if p < 0.65: return pos
        return pos + 1

    def _compute_reward(self, positions, action_bits):
        reward = 0.0
        all_at_goal, all_swim_up = True, True
        goal = self.num_locations - 1
        for i in range(self.num_rivers):
            pos, act = positions[i], action_bits[i]
            if pos == 0 and act == 0: reward += 0.1
            if pos == goal and act == 1: reward += 1.0
            if pos != goal: all_at_goal = False
            if act != 1: all_swim_up = False
        if all_at_goal and all_swim_up: reward += float(self.num_rivers)
        return reward / (2.0 * self.num_rivers)


    def get_procedural_dynamics(self):
        params = (self.num_rivers, self.num_locations, self.time_limit)
        return river_swim_step, river_swim_rollout, params

class FourRoomsEnv(gym.Env):
    def __init__(self, n=5, time_limit=50, slippery=True, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.n = n
        self.grid_size = 2 * n + 1
        self.time_limit = time_limit
        self.slippery = slippery
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size * (time_limit + 1))
        
        # Initialize goal and doors here to make environment stationary for its lifetime
        self.np_random, _ = gym.utils.seeding.np_random(kwargs.get("seed", None))
        self.doors = [
            self.n * self.grid_size + self.np_random.integers(0, self.n),
            self.n * self.grid_size + (self.n + 1 + self.np_random.integers(0, self.n)),
            self.np_random.integers(0, self.n) * self.grid_size + self.n,
            (self.n + 1 + self.np_random.integers(0, self.n)) * self.grid_size + self.n
        ]
        idx_doors = set(self.doors)
        candidates = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if not (x == self.n or y == self.n) and (y * self.grid_size + x) not in idx_doors:
                    candidates.append((x, y))
        idx1, idx2 = self.np_random.integers(len(candidates), size=2)
        while idx1 == idx2: idx2 = self.np_random.integers(len(candidates))
        self.start_x, self.start_y = candidates[idx1]
        self.goal_x, self.goal_y = candidates[idx2]
        
        self.reset()

    def get_fingerprint(self):
        return ("FourRooms", self.n, self.grid_size, self.time_limit, self.slippery, tuple(self.doors), self.goal_x, self.goal_y)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.x, self.y = self.start_x, self.start_y
        self.time = 0
        self.s = self.get_state()
        return self.s, {}

    def _is_wall(self, x, y):
        idx = y * self.grid_size + x
        return (x == self.n or y == self.n) and (idx not in self.doors)

    def get_state(self):
        return int((self.y * self.grid_size + self.x) * (self.time_limit + 1) + self.time)

    def set_state(self, state_idx):
        self.s = int(state_idx)
        self.time = self.s % (self.time_limit + 1)
        pos = self.s // (self.time_limit + 1)
        self.x, self.y = pos % self.grid_size, pos // self.grid_size

    def step(self, action):
        if self.slippery:
            p = self.np_random.random()
            if p < 0.25: eff_action = (action + 3) % 4
            elif p < 0.75: eff_action = action
            else: eff_action = (action + 1) % 4
        else: eff_action = action
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][eff_action]
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and not self._is_wall(nx, ny):
            self.x, self.y = nx, ny
        self.time += 1
        reward = (1.0 - 0.9 * (self.time / self.time_limit)) if (self.x == self.goal_x and self.y == self.goal_y) else 0.0
        terminated = (self.x == self.goal_x and self.y == self.goal_y) or (self.time >= self.time_limit)
        self.s = self.get_state()
        return self.s, reward, terminated, False, {}


    def get_procedural_dynamics(self):
        doors_arr = np.array(self.doors, dtype=np.int32)
        params = (self.n, self.grid_size, self.time_limit, self.slippery, doors_arr, self.goal_x, self.goal_y)
        return four_rooms_step, four_rooms_rollout, params

class PassengerGridEnv(gym.Env):
    def __init__(self, time_limit=70, slippery=True, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.width, self.height = 7, 6
        self.time_limit, self.slippery = time_limit, slippery
        self.num_passengers = 3
        self.passenger_positions = [(1, 2), (0, 5), (6, 4)]
        self.goal_pos = (6, 0)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.width * self.height * (1 << self.num_passengers) * (time_limit + 1))
        self.reset()

    def get_fingerprint(self):
        return ("PassengerGrid", self.width, self.height, self.num_passengers, self.time_limit, self.slippery)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y, self.mask, self.time = 0, 0, 0, 0
        self.s = self.get_state()
        return self.s, {}

    def get_state(self):
        spatial_id = self.y * self.width + self.x
        encoded = (spatial_id << self.num_passengers) | self.mask
        return int(encoded * (self.time_limit + 1) + self.time)

    def set_state(self, state_idx):
        self.s = int(state_idx)
        self.time = self.s % (self.time_limit + 1)
        encoded = self.s // (self.time_limit + 1)
        self.mask = encoded & ((1 << self.num_passengers) - 1)
        spatial_id = encoded >> self.num_passengers
        self.x, self.y = spatial_id % self.width, spatial_id // self.width

    def step(self, action):
        if self.slippery:
            p = self.np_random.random()
            if p < 0.25: eff_action = (action + 3) % 4
            elif p < 0.75: eff_action = action
            else: eff_action = (action + 1) % 4
        else: eff_action = action
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][eff_action]
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height: self.x, self.y = nx, ny
        for i, (px, py) in enumerate(self.passenger_positions):
            if self.x == px and self.y == py: self.mask |= (1 << i)
        self.time += 1
        reward = [0.0, 1.0, 3.0, 7.0][bin(self.mask).count('1')] if (self.x == self.goal_pos[0] and self.y == self.goal_pos[1]) else 0.0
        terminated = (self.x == self.goal_pos[0] and self.y == self.goal_pos[1]) or (self.time >= self.time_limit)
        self.s = self.get_state()
        return self.s, reward, terminated, False, {}


    def get_procedural_dynamics(self):
        pass_pos_arr = np.array(self.passenger_positions, dtype=np.int32)
        params = (self.width, self.height, self.time_limit, self.slippery, pass_pos_arr, self.goal_pos[0], self.goal_pos[1])
        return passenger_grid_step, passenger_grid_rollout, params

class SysadminRingEnv(gym.Env):
    def __init__(self, num_computers=20, time_limit=50, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.num_computers, self.time_limit = num_computers, time_limit
        self.action_space = spaces.Discrete(num_computers + 1)
        self.observation_space = spaces.Discrete((1 << num_computers) * (time_limit + 1))
        self.reset()

    def get_fingerprint(self):
        return ("SysadminRing", self.num_computers, self.time_limit)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mask, self.time = 0, 0
        self.s = self.get_state()
        return self.s, {}

    def get_state(self):
        return int(self.mask * (self.time_limit + 1) + self.time)

    def set_state(self, state_idx):
        self.s = int(state_idx)
        self.time = self.s % (self.time_limit + 1)
        self.mask = self.s // (self.time_limit + 1)

    def step(self, action):
        next_mask = 0
        for i in range(self.num_computers):
            if action == i: next_mask |= (1 << i)
            else:
                prev_running = (self.mask >> ((i - 1 + self.num_computers) % self.num_computers)) & 1
                self_running = (self.mask >> i) & 1
                p = [0.0238, 0.0475, 0.525, 0.95][(self_running << 1) | prev_running]
                if self.np_random.random() < p: next_mask |= (1 << i)
        self.mask, self.time = next_mask, self.time + 1
        reward = bin(self.mask).count('1') / self.num_computers
        self.s = self.get_state()
        return self.s, reward, self.time >= self.time_limit, False, {}


    def get_procedural_dynamics(self):
        probs_arr = np.array([0.0238, 0.0475, 0.525, 0.95], dtype=np.float64)
        params = (self.num_computers, self.time_limit, probs_arr)
        return sysadmin_ring_step, sysadmin_ring_rollout, params
