import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Tuple, Dict, List, Optional, Union

class SailingEnv(gym.Env):
    """
    Sailing Environment: A 2D grid world where a boat (agent) moves from (0,0) to (grid_size-1, grid_size-1).
    The movement cost depends on the alignment between the boat's direction and the wind direction.
    
    Wind follows a Markov process:
    - Initial wind: DOWN_RIGHT (1)
    - Next wind: same (p), CW (1-p)/2, CCW (1-p)/2
    
    Observation: Only the position (x, y) is observable, encoded as y * grid_size + x.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size: int = 10, wind_keep_probability: float = 0.8, time_limit: int = 100, render_mode: Optional[str] = None):
        super().__init__()
        self.grid_size = grid_size
        self.wind_keep_probability = wind_keep_probability
        self.time_limit = time_limit
        self.render_mode = render_mode
        
        # 8 directions: 0:R, 1:DR, 2:D, 3:DL, 4:L, 5:UL, 6:U, 7:UR
        self.action_space = spaces.Discrete(8)
        
        # Observation space: full state space (x, y, wind) for solver
        self.observation_space = spaces.Discrete(grid_size * grid_size * 8)
        
        # Action to (dx, dy) mapping
        self.action_to_direction = {
            0: (1, 0),   # RIGHT
            1: (1, 1),   # DOWN_RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 1),  # DOWN_LEFT
            4: (-1, 0),  # LEFT
            5: (-1, -1), # UP_LEFT
            6: (0, -1),  # UP
            7: (1, -1)   # UP_RIGHT
        }
        
        # State variables
        self.x = 0
        self.y = 0
        self.time = 0
        self.wind_dir = 1 # Initial: DOWN_RIGHT
        self.wind_sequence = []
        
        # Precompute P dynamics table for solvers (as a property/attribute)
        self._P = None

    @property
    def P(self):
        """
        Transition dynamics for solvers. 
        State space: (y * grid_size + x) * 8 + wind_dir
        Total states: grid_size * grid_size * 8
        """
        if self._P is not None:
            return self._P
            
        num_states = self.grid_size * self.grid_size * 8
        P = {s: {a: [] for a in range(8)} for s in range(num_states)}
        
        p = self.wind_keep_probability
        p_drift = (1 - p) / 2
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                for w in range(8):
                    s_idx = (y * self.grid_size + x) * 8 + w
                    for a in range(8):
                        # Movement
                        dx, dy = self.action_to_direction[a]
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                            nx, ny = x, y # No-op
                            
                        # Reward (based on current wind w)
                        diff = abs(a - w)
                        min_diff = min(diff, 8 - diff)
                        reward = -0.5 * min_diff
                        
                        # Terminal check
                        terminated = (nx == self.grid_size - 1) and (ny == self.grid_size - 1)
                        
                        # Transitions (wind drift)
                        # Outcomes: (prob, next_state, reward, terminated)
                        # 1. Wind stays same (reward based on same wind)
                        ns_keep = (ny * self.grid_size + nx) * 8 + w
                        diff_keep = abs(a - w)
                        min_diff_keep = min(diff_keep, 8 - diff_keep)
                        reward_keep = -0.5 * min_diff_keep
                        P[s_idx][a].append((p, ns_keep, reward_keep, terminated))
                        
                        # 2. Wind CW (reward based on NEW wind)
                        w_cw = (w + 1) % 8
                        ns_cw = (ny * self.grid_size + nx) * 8 + w_cw
                        diff_cw = abs(a - w_cw)
                        min_diff_cw = min(diff_cw, 8 - diff_cw)
                        reward_cw = -0.5 * min_diff_cw
                        P[s_idx][a].append((p_drift, ns_cw, reward_cw, terminated))
                        
                        # 3. Wind CCW (reward based on NEW wind)
                        w_ccw = (w - 1) % 8
                        ns_ccw = (ny * self.grid_size + nx) * 8 + w_ccw
                        diff_ccw = abs(a - w_ccw)
                        min_diff_ccw = min(diff_ccw, 8 - diff_ccw)
                        reward_ccw = -0.5 * min_diff_ccw
                        P[s_idx][a].append((p_drift, ns_ccw, reward_ccw, terminated))
                        
        self._P = P
        return self._P

    def _get_obs(self) -> int:
        return self.y * self.grid_size + self.x

    def _get_info(self) -> Dict[str, Any]:
        return {
            "position": (self.x, self.y),
            "wind_dir": self.wind_dir,
            "time": self.time
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.x = 0
        self.y = 0
        self.time = 0
        self.wind_dir = 1
        
        # Generate full wind sequence for the episode duration
        # We generate time_limit + 1 directions to be safe
        self.wind_sequence = [1]
        current_w = 1
        for _ in range(self.time_limit):
            r = self.np_random.random()
            if r < self.wind_keep_probability:
                pass # Keep
            elif r < self.wind_keep_probability + (1 - self.wind_keep_probability) / 2:
                current_w = (current_w + 1) % 8 # CW
            else:
                current_w = (current_w - 1) % 8 # CCW
            self.wind_sequence.append(current_w)
            
        self.wind_dir = self.wind_sequence[0]
        self._update_s()
        
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if self.time >= self.time_limit:
            raise RuntimeError("Step called after time limit reached.")
            
        # 1. Movement logic (no-op if out of bounds)
        dx, dy = self.action_to_direction[action]
        nx, ny = self.x + dx, self.y + dy
        
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            self.x, self.y = nx, ny
        # else: stay in place (self.x, self.y)
        
        # 2. Update time and wind FIRST
        self.time += 1
        if self.time < len(self.wind_sequence):
            self.wind_dir = self.wind_sequence[self.time]
        
        # 3. Reward calculation based on NEW wind
        diff = abs(action - self.wind_dir)
        min_diff = min(diff, 8 - diff)
        reward = -0.5 * min_diff
        
        # 4. Termination
        terminated = (self.x == self.grid_size - 1) and (self.y == self.grid_size - 1)
        truncated = (self.time >= self.time_limit)
        
        self._update_s()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _update_s(self):
        """Update the internal state representation used for get_state/set_state."""
        self.s = (self.y * self.grid_size + self.x) * 8 + self.wind_dir

    def get_state(self) -> int:
        """Returns the full state index (x, y, wind)."""
        return (self.y * self.grid_size + self.x) * 8 + self.wind_dir

    def set_state(self, state: int):
        """Restores the state from an integer index."""
        self.wind_dir = state % 8
        pos_idx = state // 8
        self.x = pos_idx % self.grid_size
        self.y = pos_idx // self.grid_size
        # Note: time and wind_sequence are NOT restored from the integer index.
        # This is fine for search, but for real transitions, they should stay consistent.

    def render(self):
        # Basic ANSI representation
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.grid_size - 1][self.grid_size - 1] = "G"
        grid[self.y][self.x] = "B"
        
        res = f"Step: {self.time}, Wind: {self.wind_dir}\n"
        for row in grid:
            res += " ".join(row) + "\n"
        return res

    def get_baseline_reward(self, seed: int) -> float:
        """
        Calculates the total reward for the 'Always DOWN_RIGHT' baseline strategy
        for a specific seed.
        """
        # Save current state to restore later
        old_state = (self.x, self.y, self.wind_dir, self.time, self.wind_sequence)
        
        self.reset(seed=seed)
        total_r = 0.0
        terminated = False
        while not terminated:
            _, r, terminated, _, _ = self.step(1) # Always Move DOWN_RIGHT
            total_r += r
            
        # Restore state
        self.x, self.y, self.wind_dir, self.time, self.wind_sequence = old_state
        self._update_s()
        
        return total_r
