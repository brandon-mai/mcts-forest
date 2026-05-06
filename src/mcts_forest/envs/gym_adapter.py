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

    def get_numba_dynamics(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extracts transition dynamics for Toy-Text environments as NumPy arrays.
        Enables Numba-accelerated rollouts with precise probabilistic sampling.
        """
        if not hasattr(self.env, 'P'):
            return None
            
        # Check disk cache
        cache_path = None
        if hasattr(self.env, "get_fingerprint"):
            fp = str(self.env.get_fingerprint())
            h = hashlib.md5(fp.encode()).hexdigest()
            cache_path = os.path.join(".cache", f"dynamics_{h}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                return data['transitions'], data['rewards'], data['dones'], data['probs_cum']

        ns = self.env.observation_space.n
        na = self.env.action_space.n
        
        # Determine max outcomes to create fixed-shape arrays
        max_outcomes = 0
        for s in range(ns):
            for a in range(na):
                max_outcomes = max(max_outcomes, len(self.env.P[s][a]))
        
        if max_outcomes == 0:
            return None

        # [states, actions, outcomes]
        transitions = np.zeros((ns, na, max_outcomes), dtype=np.int32)
        rewards = np.zeros((ns, na, max_outcomes), dtype=np.float32)
        dones = np.zeros((ns, na, max_outcomes), dtype=np.bool_)
        probs_cum = np.zeros((ns, na, max_outcomes), dtype=np.float32)
        
        for s in range(ns):
            for a in range(na):
                outcomes = self.env.P[s][a]
                cum_p = 0.0
                
                # Handle states with no defined transitions (e.g., terminal states in some JAIR envs)
                if not outcomes:
                    for i in range(max_outcomes):
                        transitions[s, a, i] = s
                        rewards[s, a, i] = 0.0
                        dones[s, a, i] = True
                        probs_cum[s, a, i] = 1.0
                    continue

                for i in range(max_outcomes):
                    target_idx = i if i < len(outcomes) else (len(outcomes) - 1)
                    prob, next_s, r, done = outcomes[target_idx]
                    
                    if i < len(outcomes):
                        cum_p += prob
                        probs_cum[s, a, i] = cum_p
                    else:
                        probs_cum[s, a, i] = 1.0 # Padding
                        
                    transitions[s, a, i] = next_s
                    rewards[s, a, i] = (r + self.reward_offset) * self.reward_scale
                    dones[s, a, i] = done
        
        if cache_path:
            os.makedirs(".cache", exist_ok=True)
            np.savez_compressed(cache_path, transitions=transitions, rewards=rewards, dones=dones, probs_cum=probs_cum)

        return transitions, rewards, dones, probs_cum
