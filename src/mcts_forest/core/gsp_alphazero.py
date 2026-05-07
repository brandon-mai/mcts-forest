import numpy as np
import math
import numba
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any, List

class GSPAlphaZeroNet(nn.Module):
    def __init__(self, obs_dim=16, n_actions=4, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.fc(x)
        return self.policy_head(x), torch.sigmoid(self.value_head(x))

@njit(cache=True)
def gsp_alphazero_core(n_sim, horizon, initial_state, c_puct, gamma, p, step_fn, params, n_actions, visit_count, q_hat, v_hat, action_visits, priors, state_h_to_node, node_states, node_counter, model_v, model_p):
    for _ in range(n_sim):
        curr_node = np.int32(0)
        curr_h = 0
        p_nodes, p_a, p_r = np.empty(horizon+1, dtype=np.int32), np.empty(horizon+1, dtype=np.int32), np.empty(horizon+1, dtype=np.float32)
        p_len, v_leaf = 0, 0.0
        
        while curr_h < horizon:
            s_idx = node_states[curr_node]
            n_p = visit_count[curr_node]
            sqrt_n_p = math.sqrt(n_p) if n_p > 0 else 1.0
            
            best_val, best_a = -1e18, -1
            for a in range(n_actions):
                score = q_hat[curr_node, a] + c_puct * priors[curr_node, a] * sqrt_n_p / (1 + action_visits[curr_node, a])
                if score > best_val: best_val, best_a = score, a
            
            a = best_a
            # Use procedural step
            next_s, r, done = step_fn(s_idx, a, *params)
            
            s_key = (np.int32(next_s), np.int32(curr_h + 1))
            if s_key in state_h_to_node:
                next_node = np.int32(state_h_to_node[s_key])
            else:
                next_node = np.int32(-1)
            
            p_nodes[p_len], p_a[p_len], p_r[p_len] = curr_node, a, r
            p_len += 1
            if done: break
            if next_node == -1:
                new_id = node_counter[0]; node_counter[0] += 1
                state_h_to_node[s_key] = new_id
                node_states[new_id] = next_s
                v_leaf = model_v[np.int32(next_s)]
                priors[new_id] = model_p[np.int32(next_s)]
                v_hat[new_id], visit_count[new_id] = v_leaf, 0
                break
            curr_node, curr_h = next_node, curr_h + 1
        else: v_leaf = v_hat[curr_node]

        v_back = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r = p_nodes[i], p_a[i], p_r[i]
            action_visits[n_id, a] += 1
            visit_count[n_id] += 1
            q_hat[n_id, a] += (r + gamma * v_back - q_hat[n_id, a]) / action_visits[n_id, a]
            
            q_min = 1e18
            for act in range(n_actions):
                if action_visits[n_id, act] > 0 and q_hat[n_id, act] < q_min:
                    q_min = q_hat[n_id, act]
            curr_shift = max(0.0, -q_min) + 1.0
            
            w_sum, n_node = 0.0, float(visit_count[n_id])
            for act in range(n_actions):
                if action_visits[n_id, act] > 0:
                    q_val = q_hat[n_id, act] + curr_shift
                    if q_val < 0.0: q_val = 0.0
                    w_sum += (action_visits[n_id, act] / n_node) * (q_val ** p)
            v_hat[n_id] = (w_sum ** (1.0 / p)) - curr_shift
            v_back = v_hat[n_id]

class GSPAlphaZero:
    def __init__(self, env, model_adapter, c_puct=1.25, p=2.0, gamma=0.99, simulation_limit=100, **kwargs):
        self.env, self.model = env, model_adapter.model.to(model_adapter.device)
        self.c_puct, self.p, self.gamma, self.simulation_limit = c_puct, p, gamma, simulation_limit
        
        # Enforce procedural dynamics
        try:
            self.step_fn, _, self.params = env.get_procedural_dynamics()
        except (AttributeError, NotImplementedError) as e:
            raise RuntimeError(f"GSPAlphaZero requires procedural dynamics: {e}")
            
        self.obs_dim = env.observation_space.n
        self.n_actions = env.action_space_size
        self.horizon = 100

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        max_nodes = self.simulation_limit * self.horizon + 1
        n_actions = self.n_actions
        v_count = np.zeros(max_nodes, dtype=np.int32); q_hat = np.zeros((max_nodes, n_actions), dtype=np.float32)
        v_hat = np.zeros(max_nodes, dtype=np.float32); n_states = np.zeros(max_nodes, dtype=np.int32)
        n_counter = np.array([1], dtype=np.int32); a_visits = np.zeros((max_nodes, n_actions), dtype=np.int32)
        priors = np.zeros((max_nodes, n_actions), dtype=np.float32)
        state_h_to_node = NumbaDict.empty(key_type=numba.types.Tuple((numba.int32, numba.int32)), value_type=numba.types.int32)
        state_h_to_node[(np.int32(initial_state), np.int32(0))] = np.int32(0)
        n_states[0] = initial_state
        model_v, model_p = self._get_predictions()
        v_hat[0], priors[0] = model_v[initial_state], model_p[initial_state]
        
        gsp_alphazero_core(self.simulation_limit, self.horizon, initial_state, self.c_puct, self.gamma, self.p, 
                          self.step_fn, self.params, n_actions,
                          v_count, q_hat, v_hat, a_visits, priors, state_h_to_node, n_states, n_counter, model_v, model_p)
        return int(np.argmax(a_visits[0])), {"action_visits": a_visits[0].copy()}

    def _get_predictions(self):
        with torch.no_grad():
            s_oh = torch.eye(self.obs_dim).to(next(self.model.parameters()).device)
            p, v = self.model(s_oh)
            return v.cpu().numpy().flatten(), F.softmax(p, dim=-1).cpu().numpy()
