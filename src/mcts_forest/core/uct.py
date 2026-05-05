import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
from mcts_forest.core.base import sample_discrete_transition, random_rollout_discrete
from mcts_forest.utils.visualizer import SurrogateNode


@njit(cache=True)
def uct_core(n_sim, horizon, c, gamma, rollout_limit, dynamics, visit_count, q_hat, action_visits, child_nodes, node_states, node_counter, reward_offset, reward_scale):
    transitions, rewards, dones, probs_cum = dynamics
    p_nodes = np.empty(horizon + 1, dtype=np.int32)
    p_a = np.empty(horizon + 1, dtype=np.int32)
    p_r_int = np.empty(horizon + 1, dtype=np.float32)
    
    for _ in range(n_sim):
        curr_node, curr_h = 0, 0
        p_len = 0
        v_leaf = 0.0
        while curr_h < horizon:
            s_idx = node_states[curr_node]
            best_val, best_a, n_p = -1e18, -1, visit_count[curr_node]
            ln_p = math.log(n_p) if n_p > 0 else 0.0
            for a in range(q_hat.shape[1]):
                n_sa = action_visits[curr_node, a]
                if n_sa == 0:
                    best_a = a
                    break
                val = q_hat[curr_node, a] + c * math.sqrt(ln_p / n_sa)
                if val > best_val:
                    best_val, best_a = val, a
            
            a = best_a
            next_s, r, done = sample_discrete_transition(s_idx, a, transitions, rewards, dones, probs_cum)
            
            # Internal normalization
            r_int = (r + reward_offset) * reward_scale
            
            key = (np.int32(curr_node), np.int32(a))
            
            if key in child_nodes:
                next_node = child_nodes[key]
            else:
                next_node = -1
                
            p_nodes[p_len], p_a[p_len], p_r_int[p_len] = curr_node, a, r_int
            p_len += 1
            
            if done:
                v_leaf = 0.0
                break
            if next_node == -1:
                new_id = node_counter[0]
                node_counter[0] += 1
                child_nodes[key], node_states[new_id] = new_id, next_s
                v_leaf = random_rollout_discrete(next_s, transitions, rewards, dones, probs_cum, rollout_limit, gamma, reward_offset, reward_scale)
                visit_count[new_id] = 1
                break
            curr_node, curr_h = next_node, curr_h + 1
        else:
            v_leaf = random_rollout_discrete(node_states[curr_node], transitions, rewards, dones, probs_cum, rollout_limit, gamma, reward_offset, reward_scale)

        v_nxt = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r_int = p_nodes[i], p_a[i], p_r_int[i]
            action_visits[n_id, a] += 1
            nv = action_visits[n_id, a]
            q_hat[n_id, a] += (r_int + gamma * v_nxt - q_hat[n_id, a]) / nv
            visit_count[n_id] += 1
            v_nxt = r_int + gamma * v_nxt

class UCT:
    def __init__(self, env, c=1.0, horizon=100, gamma=0.99, rollout_limit=100, simulation_limit=1000, 
                 internal_reward_offset=0.0, internal_reward_scale=1.0, **kwargs):
        self.env, self.c, self.horizon, self.gamma, self.rollout_limit, self.simulation_limit = env, c, horizon, gamma, rollout_limit, simulation_limit
        self.reward_offset = internal_reward_offset
        self.reward_scale = internal_reward_scale
        self.dynamics = env.get_numba_dynamics()

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        max_n = self.simulation_limit + 1
        v_count, q_hat, a_visits = np.zeros(max_n, dtype=np.int32), np.zeros((max_n, self.dynamics[0].shape[1]), dtype=np.float32), np.zeros((max_n, self.dynamics[0].shape[1]), dtype=np.int32)
        n_states, n_counter = np.zeros(max_n, dtype=np.int32), np.array([1], dtype=np.int32)
        n_states[0], child_nodes = initial_state, NumbaDict.empty(key_type=numba.types.Tuple((numba.int32, numba.int32)), value_type=numba.int32)
        uct_core(self.simulation_limit, self.horizon, self.c, self.gamma, self.rollout_limit, self.dynamics, v_count, q_hat, a_visits, child_nodes, n_states, n_counter, self.reward_offset, self.reward_scale)
        
        return int(np.argmax(q_hat[0])), {"root": None, "root_v": float(np.max(q_hat[0]))}

    def get_name(self) -> str:
        return f"uct_sim{self.simulation_limit}"
