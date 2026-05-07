import gymnasium as gym
import numpy as np
import copy
import pickle
import hashlib
import os
from typing import Any, Tuple, Dict, List, Protocol, TypeVar, Optional
from mcts_forest.envs.base import EnvBase

class GymAdapter(EnvBase):
    """
    An adapter for Gymnasium environments that satisfies the MCTSCompatibleEnv protocol.
    """
    def __init__(self, env_or_id: Any, render_mode: Optional[str] = None, **kwargs):
        self.is_slippery = kwargs.get("is_slippery", False)
        self.is_rainy = kwargs.get("is_rainy", False)
        self.map_name = kwargs.get("map_name", None)
        
        # Remove seed from kwargs if present
        kwargs.pop("seed", None)
        kwargs.pop("episode_idx", None)
        self.reward_offset = kwargs.pop("reward_offset", 0.0)
        self.reward_scale = kwargs.pop("reward_scale", 1.0)
        
        if isinstance(env_or_id, str):
            self.env_id = env_or_id
            full_env = gym.make(env_or_id, render_mode=render_mode, **kwargs)
            self.env = full_env.unwrapped
        else:
            self.env_id = "custom_env"
            self.env = env_or_id
        
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.terminal_states = set()
        
        # Pre-identify terminal states for Toy-Text envs for efficiency
        if hasattr(self.env, 'P'):
            for s in range(self.env.observation_space.n):
                # If all actions from state s lead to terminated=True, it's terminal.
                is_term = True
                for a in range(self.env.action_space.n):
                    # P[s][a] is list of (prob, next_s, reward, terminated)
                    for prob, next_s, reward, terminated in self.env.P[s][a]:
                        if not terminated:
                            is_term = False
                            break
                    if not is_term: break
                if is_term:
                    self.terminal_states.add(s)

    def get_name(self) -> str:
        """Returns a cleaned environment name, e.g., frozenlake_slip, taxi."""
        import re
        # Remove versioning -v0, -v1, etc.
        name = re.sub(r'-v\d+$', '', self.env_id).lower()
        
        if "taxi" in name:
            if self.is_rainy:
                name += "_rain"
                # If rainy_probability is set and not default (0.8), add it to name
                prob = getattr(self.env, "rainy_probability", 0.8)
                if abs(prob - 0.8) > 1e-6:
                    name += f"_{prob:.1f}"
            return name
        
        if "frozenlake" in name:
            # Handle board size and slippery status
            name = "frozenlake"
            if self.map_name and self.map_name != "4x4":
                name += self.map_name.lower()
            elif hasattr(self.env, 'nrow') and self.env.nrow == 8:
                name += "8x8"
                
            if self.is_slippery:
                name += "_slip"
                # If we have a custom slip probability (though FL-v1 doesn't officially support it as a kwarg yet)
                prob = getattr(self.env, "slip_probability", None)
                if prob is not None:
                    name += f"_{prob:.1f}"
            return name
            
        return name

    def reset(self, seed: Optional[int] = None) -> Any:
        obs, _ = self.env.reset(seed=seed)
        return self.get_state()

    def get_state(self) -> Any:
        """
        Snapshot state. For envs with .s (Toy-Text/Custom), return it.
        Otherwise, use pickle for full isolation.
        """
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 's'):
            return self.env.unwrapped.s
        if hasattr(self.env, 's'):
            return self.env.s
        return pickle.dumps(self.env)

    def set_state(self, state: Any):
        """
        Restore state.
        """
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 's'):
            self.env.unwrapped.s = state
        elif hasattr(self.env, 's'):
            self.env.s = state
        elif isinstance(state, bytes):
            self.env = pickle.loads(state)
        else:
            self.env = copy.deepcopy(state)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        # Gymnasium returns obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = (reward + self.reward_offset) * self.reward_scale
        return self.get_state(), reward, terminated, truncated, info

    def get_legal_actions(self, state: Any) -> List[int]:
        # If the environment has its own legal actions logic, use it
        if hasattr(self.env, 'get_legal_actions'):
            return self.env.get_legal_actions(state)
            
        # Fallback for standard Gym envs
        if state in self.terminal_states:
            return []
        if isinstance(self.action_space, gym.spaces.Discrete):
            return list(range(self.action_space.n))
        return []

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    @property
    def action_space_size(self) -> int:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n
        return 0

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    def get_procedural_dynamics(self):
        """
        Returns procedural dynamics functions and parameters.
        Only supported for environments with procedural implementations.
        """
        # 1. Check if the underlying env has it (JAIR envs)
        if hasattr(self.env, "get_procedural_dynamics"):
            return self.env.get_procedural_dynamics()
            
        # 2. Hardcoded support for standard Toy-Text envs
        env_name = self.get_name()
        
        if "frozenlake" in env_name:
            from mcts_forest.envs.procedural import frozen_lake_step, frozen_lake_rollout
            desc = self.env.unwrapped.desc
            # Convert desc to integer grid: F=0, H=1, G=2, S=3
            h, w = desc.shape
            grid = np.zeros(h * w, dtype=np.int32)
            mapping = {b'F': 0, b'H': 1, b'G': 2, b'S': 3}
            for i in range(h):
                for j in range(w):
                    grid[i * w + j] = mapping[desc[i, j]]
            params = (grid, w, h, self.is_slippery)
            return frozen_lake_step, frozen_lake_rollout, params
            
        if "taxi" in env_name:
            from mcts_forest.envs.procedural import taxi_step, taxi_rollout
            prob = getattr(self.env, "rainy_probability", 0.0) if self.is_rainy else 0.0
            params = (float(prob),)
            return taxi_step, taxi_rollout, params
            
        raise NotImplementedError(f"Procedural dynamics not implemented for environment: {env_name}")
