import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
from mcts_forest.core.base import sample_discrete_transition, random_rollout_discrete
from mcts_forest.utils.visualizer import SurrogateNode

@njit(cache=True)
def sp_uct_core(n_sim, horizon, c, gamma, p, rollout_limit, dynamics, T_s, Q_hat, V_hat, T_sa, child_nodes, node_states, node_counter, reward_offset, reward_scale):
    transitions, rewards, dones, probs_cum = dynamics
    for _ in range(n_sim):
        curr_node, curr_h = 0, 0
        p_nodes, p_a, p_r, p_next_s = np.empty(101, dtype=np.int32), np.empty(101, dtype=np.int32), np.empty(101, dtype=np.float32), np.empty(101, dtype=np.int32)
        p_len = 0
        v_leaf = 0.0
        
        while curr_h < horizon:
            s_idx = node_states[curr_node]
            best_val, best_a, n_p = -1e18, -1, T_s[curr_node]
            
            for a in range(Q_hat.shape[1]):
                n_sa = T_sa[curr_node, a]
                if n_sa == 0:
                    best_a = a
                    break
                bonus = c * (n_p ** 0.25 / n_sa ** 0.5)
                val = Q_hat[curr_node, a] + bonus
                if val > best_val:
                    best_val, best_a = val, a
            
            a = best_a
            next_s, r, done = sample_discrete_transition(s_idx, a, transitions, rewards, dones, probs_cum)
            
            # Internal normalization
            r_int = (r + reward_offset) * reward_scale
            
            key = (np.int32(curr_node), np.int32(a), np.int32(next_s))
            if key in child_nodes:
                next_node = child_nodes[key]
            else:
                next_node = -1
            
            p_nodes[p_len], p_a[p_len], p_r[p_len], p_next_s[p_len] = curr_node, a, r_int, next_s
            p_len += 1
            
            if done:
                v_leaf = 0.0
                break
            if next_node == -1:
                # Expand Node
                new_id = node_counter[0]
                node_counter[0] += 1
                child_nodes[key], node_states[new_id] = new_id, next_s
                v_leaf_raw = random_rollout_discrete(next_s, transitions, rewards, dones, probs_cum, rollout_limit, gamma)
                # Simplified leaf normalization:
                v_leaf = (v_leaf_raw + reward_offset / (1-gamma)) * reward_scale if gamma < 1.0 else (v_leaf_raw + reward_offset * rollout_limit) * reward_scale
                
                V_hat[new_id] = v_leaf
                T_s[new_id] = 1
                break
            curr_node, curr_h = next_node, curr_h + 1
        else:
            v_leaf_raw = random_rollout_discrete(node_states[curr_node], transitions, rewards, dones, probs_cum, rollout_limit, gamma)
            v_leaf = (v_leaf_raw + reward_offset / (1-gamma)) * reward_scale if gamma < 1.0 else (v_leaf_raw + reward_offset * rollout_limit) * reward_scale

        # Backpropagation (SimulateV/Q recursive equivalent)
        v_back = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r_int = p_nodes[i], p_a[i], p_r[i]
            T_sa[n_id, a] += 1
            nv = T_sa[n_id, a]
            Q_hat[n_id, a] += (r_int + gamma * v_back - Q_hat[n_id, a]) / nv
            n_p = T_s[n_id] + 1
            T_s[n_id] = n_p
            
            q_min = 1e18
            for act in range(Q_hat.shape[1]):
                if T_sa[n_id, act] > 0 and Q_hat[n_id, act] < q_min:
                    q_min = Q_hat[n_id, act]
            
            curr_shift = max(0.0, -q_min) + 1.0
            weighted_sum = 0.0
            for act in range(Q_hat.shape[1]):
                n_act = T_sa[n_id, act]
                if n_act > 0:
                    q_val = Q_hat[n_id, act] + curr_shift
                    if q_val < 0.0: q_val = 0.0
                    weighted_sum += (n_act / n_p) * (q_val ** p)
            
            V_hat[n_id] = (weighted_sum ** (1.0 / p)) - curr_shift
            v_back = V_hat[n_id]

class SPUCT:
    def __init__(self, env, c=1.0, p=2.0, horizon=100, gamma=0.99, rollout_limit=100, simulation_limit=1000, 
                 internal_reward_offset=0.0, internal_reward_scale=1.0, init_q=0.0, **kwargs):
        self.env, self.c, self.p, self.horizon, self.gamma, self.rollout_limit, self.simulation_limit = env, c, p, horizon, gamma, rollout_limit, simulation_limit
        self.reward_offset = internal_reward_offset
        self.reward_scale = internal_reward_scale
        self.init_q = init_q
        self.dynamics = env.get_numba_dynamics()

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        max_n = self.simulation_limit * 3 + 1
        T_s, Q_hat, V_hat = np.zeros(max_n, dtype=np.int32), np.full((max_n, self.dynamics[0].shape[1]), self.init_q, dtype=np.float32), np.full(max_n, self.init_q, dtype=np.float32)
        T_sa, n_states, n_counter = np.zeros((max_n, self.dynamics[0].shape[1]), dtype=np.int32), np.zeros(max_n, dtype=np.int32), np.array([1], dtype=np.int32)
        n_states[0], child_nodes = initial_state, NumbaDict.empty(key_type=numba.types.Tuple((numba.int32, numba.int32, numba.int32)), value_type=numba.int32)
        
        sp_uct_core(self.simulation_limit, self.horizon, self.c, self.gamma, self.p, self.rollout_limit, self.dynamics, T_s, Q_hat, V_hat, T_sa, child_nodes, n_states, n_counter, self.reward_offset, self.reward_scale)

        return int(np.argmax(Q_hat[0])), {"root": None, "root_v": float(V_hat[0])}

    def get_name(self) -> str:
        return f"sp_uct_p{self.p}_sim{self.simulation_limit}"
