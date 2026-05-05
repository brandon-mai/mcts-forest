import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any, Dict, Optional
from numba import njit

@njit(cache=True)
def sample_discrete_transition(state, action, transitions, rewards, dones, probs_cum):
    num_outcomes = transitions.shape[2]
    r_val = np.random.random()
    outcome_idx = num_outcomes - 1
    for i in range(num_outcomes):
        if r_val < probs_cum[state, action, i]:
            outcome_idx = i
            break
    return transitions[state, action, outcome_idx], rewards[state, action, outcome_idx], dones[state, action, outcome_idx]

@njit(cache=True)
def random_rollout_discrete(state, transitions, rewards, dones, probs_cum, limit, gamma, reward_offset, reward_scale):
    curr_s = state
    total_r = 0.0
    curr_g = 1.0
    num_actions = transitions.shape[1]
    for _ in range(limit):
        a = np.random.randint(0, num_actions)
        next_s, r, done = sample_discrete_transition(curr_s, a, transitions, rewards, dones, probs_cum)
        # Apply internal transformation
        r_int = (r + reward_offset) * reward_scale
        total_r += curr_g * r_int
        curr_g *= gamma
        curr_s = next_s
        if done: break
    return total_r

class ExperienceBuffer:
    """Stores (state, search_policy, final_outcome) for training Zero-style algorithms."""
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experiences: List[Tuple[Any, np.ndarray, float]]):
        self.buffer.extend(experiences)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)

    def get_batches(self, batch_size: int):
        if not self.buffer: return
        indices = np.arange(len(self.buffer))
        np.random.shuffle(indices)
        for i in range(0, len(self.buffer), batch_size):
            batch_idx = indices[i:i+batch_size]
            states, policies, values = [], [], []
            for idx in batch_idx:
                s, p, v = self.buffer[idx]
                states.append(s)
                policies.append(p)
                values.append(v)
            yield np.array(states), np.array(policies), np.array(values)

class TorchModelAdapter:
    """Standardized adapter for Neural MCTS solvers."""
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
