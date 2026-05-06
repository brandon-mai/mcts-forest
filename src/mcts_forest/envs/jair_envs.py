import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class FactoredRiverSwimEnv(gym.Env):
    def __init__(self, num_rivers=4, num_locations=8, time_limit=35, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.num_rivers = num_rivers
        self.num_locations = num_locations
        self.time_limit = time_limit
        self.action_space = spaces.Discrete(1 << num_rivers)
        self.observation_space = spaces.Discrete((num_locations ** num_rivers) * (time_limit + 1))
        self._P = None
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

    @property
    def P(self):
        if self._P is not None: return self._P
        ns = self.observation_space.n
        na = self.action_space.n
        tl = self.time_limit
        P = {s: {a: [] for a in range(na)} for s in range(ns)}
        
        # Precompute river-wise outcomes for (pos, bit)
        # river_outcomes[pos][bit] = [(prob, next_pos)]
        river_outcomes = []
        for pos in range(self.num_locations):
            pos_bits = []
            for bit in [0, 1]:
                if bit == 0: outcomes = [(1.0, max(0, pos - 1))]
                elif pos == 0: outcomes = [(0.4, 0), (0.6, 1)]
                elif pos == self.num_locations - 1: outcomes = [(0.4, self.num_locations - 2), (0.6, self.num_locations - 1)]
                else: outcomes = [(0.05, pos-1), (0.60, pos), (0.35, pos+1)]
                pos_bits.append(outcomes)
            river_outcomes.append(pos_bits)

        for s in range(ns):
            t = s % (tl + 1)
            if t >= tl: continue
            
            sid = s // (tl + 1)
            temp_sid = sid
            pos_list = []
            for _ in range(self.num_rivers):
                pos_list.append(temp_sid % self.num_locations)
                temp_sid //= self.num_locations
            pos_arr = np.array(pos_list)

            for a in range(na):
                a_bits = [(a >> i) & 1 for i in range(self.num_rivers)]
                reward = self._compute_reward(pos_arr, a_bits)
                
                # Combine outcomes
                cur_outcomes = [(1.0, 0)] # (prob, next_sid)
                factor = 1
                for r in range(self.num_rivers):
                    r_out = river_outcomes[pos_list[r]][a_bits[r]]
                    next_level = []
                    for p1, s1 in cur_outcomes:
                        for p2, pos2 in r_out:
                            next_level.append((p1 * p2, s1 + pos2 * factor))
                    cur_outcomes = next_level
                    factor *= self.num_locations
                
                for prob, next_sid in cur_outcomes:
                    next_s = next_sid * (tl + 1) + (t + 1)
                    P[s][a].append((prob, next_s, reward, (t + 1 >= tl)))
        self._P = P
        return self._P

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
        self.doors = [n*self.grid_size+0, n*self.grid_size+(n+1), 0*self.grid_size+n, (n+1)*self.grid_size+n]
        self.reset()

    def get_fingerprint(self):
        return ("FourRooms", self.n, self.grid_size, self.time_limit, self.slippery)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.doors = [
            self.n * self.grid_size + self.np_random.integers(0, self.n),
            self.n * self.grid_size + (self.n + 1 + self.np_random.integers(0, self.n)),
            self.np_random.integers(0, self.n) * self.grid_size + self.n,
            (self.n + 1 + self.np_random.integers(0, self.n)) * self.grid_size + self.n
        ]
        candidates = [(x,y) for y in range(self.grid_size) for x in range(self.grid_size) if not self._is_wall(x,y)]
        idx1, idx2 = self.np_random.integers(len(candidates), size=2)
        while idx1 == idx2: idx2 = self.np_random.integers(len(candidates))
        self.x, self.y = candidates[idx1]
        self.goal_x, self.goal_y = candidates[idx2]
        self.time = 0
        self.s = self.get_state()
        self._P = None
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

    @property
    def P(self):
        if self._P is not None: return self._P
        P = {s: {a: [] for a in range(4)} for s in range(self.observation_space.n)}
        for s in range(self.observation_space.n):
            t = s % (self.time_limit + 1)
            if t >= self.time_limit: continue
            pos = s // (self.time_limit + 1)
            x, y = pos % self.grid_size, pos // self.grid_size
            if x == self.goal_x and y == self.goal_y: continue
            for a in range(4):
                if self.slippery: outcomes = [(0.25, (a+3)%4), (0.5, a), (0.25, (a+1)%4)]
                else: outcomes = [(1.0, a)]
                for prob, eff_a in outcomes:
                    dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][eff_a]
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and not self._is_wall(nx, ny)):
                        nx, ny = x, y
                    next_s = (ny * self.grid_size + nx) * (self.time_limit + 1) + (t + 1)
                    reward = (1.0 - 0.9 * ((t+1) / self.time_limit)) if (nx == self.goal_x and ny == self.goal_y) else 0.0
                    done = (nx == self.goal_x and ny == self.goal_y) or (t + 1 >= self.time_limit)
                    P[s][a].append((prob, next_s, reward, done))
        self._P = P
        return self._P

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

    @property
    def P(self):
        if hasattr(self, "_P") and self._P is not None: return self._P
        P = {s: {a: [] for a in range(4)} for s in range(self.observation_space.n)}
        for s in range(self.observation_space.n):
            t = s % (self.time_limit + 1)
            if t >= self.time_limit: continue
            encoded = s // (self.time_limit + 1)
            mask = encoded & ((1 << self.num_passengers) - 1)
            spatial_id = encoded >> self.num_passengers
            x, y = spatial_id % self.width, spatial_id // self.width
            if x == self.goal_pos[0] and y == self.goal_pos[1]: continue
            for a in range(4):
                if self.slippery: outcomes = [(0.25, (a+3)%4), (0.5, a), (0.25, (a+1)%4)]
                else: outcomes = [(1.0, a)]
                for prob, eff_a in outcomes:
                    dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][eff_a]
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < self.width and 0 <= ny < self.height): nx, ny = x, y
                    n_mask = mask
                    for i, (px, py) in enumerate(self.passenger_positions):
                        if nx == px and ny == py: n_mask |= (1 << i)
                    reward = [0.0, 1.0, 3.0, 7.0][bin(n_mask).count('1')] if (nx == self.goal_pos[0] and ny == self.goal_pos[1]) else 0.0
                    done = (nx == self.goal_pos[0] and ny == self.goal_pos[1]) or (t + 1 >= self.time_limit)
                    next_s = ((ny * self.width + nx) << self.num_passengers | n_mask) * (self.time_limit + 1) + (t + 1)
                    P[s][a].append((prob, next_s, reward, done))
        self._P = P
        return self._P

class SysadminRingEnv(gym.Env):
    def __init__(self, num_computers=10, time_limit=50, render_mode=None, **kwargs):
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
        self.mask, self.time = (1 << self.num_computers) - 1, 0
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
                p = [0.0238, 0.525, 0.0475, 0.95][(self_running << 1) | prev_running]
                if self.np_random.random() < p: next_mask |= (1 << i)
        self.mask, self.time = next_mask, self.time + 1
        reward = bin(self.mask).count('1') / self.num_computers
        self.s = self.get_state()
        return self.s, reward, self.time >= self.time_limit, False, {}

    @property
    def P(self):
        if hasattr(self, "_P") and self._P is not None: return self._P
        ns = self.observation_space.n
        na = self.action_space.n
        tl = self.time_limit
        nc = self.num_computers
        P = {s: {a: [] for a in range(na)} for s in range(ns)}
        
        for s in range(ns):
            t = s % (tl + 1)
            if t >= tl: continue
            mask = s // (tl + 1)
            
            # Precompute machine-wise r_outcomes for this mask
            all_machine_probs = []
            for i in range(nc):
                prev_machine = (i - 1 + nc) % nc
                prev_running = (mask >> prev_machine) & 1
                self_running = (mask >> i) & 1
                p_stay = [0.0238, 0.525, 0.0475, 0.95][(self_running << 1) | prev_running]
                all_machine_probs.append(p_stay)

            for a in range(na):
                # Combined probabilities of all computers
                outcomes = [(1.0, 0)] # (prob, next_mask)
                for i in range(nc):
                    if a == i: r_out = [(1.0, 1)]
                    else:
                        p_stay = all_machine_probs[i]
                        r_out = [(p_stay, 1), (1.0 - p_stay, 0)]
                    
                    next_level = []
                    for p1, m1 in outcomes:
                        for p2, bit2 in r_out:
                            next_level.append((p1 * p2, m1 | (bit2 << i)))
                    outcomes = next_level
                
                for prob, n_mask in outcomes:
                    reward = bin(n_mask).count('1') / nc
                    next_s = n_mask * (tl + 1) + (t + 1)
                    P[s][a].append((prob, next_s, reward, (t + 1 >= tl)))
        self._P = P
        return self._P
