import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any

@njit
def gsp_uct_full_core(n_sim, horizon, initial_state, c, gamma, p, rollout_limit, step_fn, rollout_fn, params, n_actions, T_s, Q_hat, V_hat, T_sa, state_to_node, node_states, node_counter, reward_offset, reward_scale):
    p_nodes = np.empty(horizon + 1, dtype=np.int32)
    p_a = np.empty(horizon + 1, dtype=np.int32)
    p_r_int = np.empty(horizon + 1, dtype=np.float32)
    
    for _ in range(n_sim):
        curr_node, curr_h = 0, 0
        p_len = 0
        v_leaf = 0.0
        
        while curr_h < horizon:
            s_idx = node_states[curr_node]
            best_val, best_a, n_p = -1e18, -1, T_s[curr_node]
            
            for a in range(n_actions):
                n_sa = T_sa[curr_node, a]
                if n_sa == 0:
                    best_a = a
                    break
                bonus = c * (n_p ** 0.25 / n_sa ** 0.5)
                val = Q_hat[curr_node, a] + bonus
                if val > best_val:
                    best_val, best_a = val, a
            
            a = best_a
            # Use procedural step
            next_s, r, done = step_fn(s_idx, a, *params)
            
            r_int = (r + reward_offset) * reward_scale
            s_key = np.int32(next_s)
            if s_key in state_to_node:
                next_node = state_to_node[s_key]
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
                state_to_node[np.int32(next_s)] = new_id
                node_states[new_id] = next_s
                # Use procedural rollout
                v_leaf = (rollout_fn(next_s, *params, rollout_limit, gamma) + reward_offset) * reward_scale
                
                V_hat[new_id] = v_leaf
                T_s[new_id] = 1
                break
            
            curr_node, curr_h = next_node, curr_h + 1
        else:
            v_leaf = (rollout_fn(node_states[curr_node], *params, rollout_limit, gamma) + reward_offset) * reward_scale

        v_back = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r_int = p_nodes[i], p_a[i], p_r_int[i]
            T_sa[n_id, a] += 1
            nv = T_sa[n_id, a]
            Q_hat[n_id, a] += (r_int + gamma * v_back - Q_hat[n_id, a]) / nv
            n_p = T_s[n_id] + 1
            T_s[n_id] = n_p
            
            q_min = 1e18
            for act in range(n_actions):
                if T_sa[n_id, act] > 0 and Q_hat[n_id, act] < q_min:
                    q_min = Q_hat[n_id, act]
            curr_shift = max(0.0, -q_min) + reward_scale
            
            if math.isinf(p):
                max_q = -1e18
                for act in range(n_actions):
                    if T_sa[n_id, act] > 0 and Q_hat[n_id, act] > max_q:
                        max_q = Q_hat[n_id, act]
                V_hat[n_id] = max_q
            else:
                weighted_sum = 0.0
                for act in range(n_actions):
                    n_act = T_sa[n_id, act]
                    if n_act > 0:
                        q_val = Q_hat[n_id, act] + curr_shift
                        if q_val < 0.0: q_val = 0.0
                        weighted_sum += (n_act / n_p) * (q_val ** p)
                V_hat[n_id] = (weighted_sum ** (1.0 / p)) - curr_shift
            v_back = V_hat[n_id]

class GSPUCTFull:
    def __init__(self, env, c=1.0, p=2.0, gamma=0.98, rollout_limit=100, simulation_limit=1000, 
                 internal_reward_offset=0.0, internal_reward_scale=1.0, init_q=0.0, **kwargs):
        self.env, self.c = env, c
        if p == 'inf':
            self.p = np.inf
        else:
            self.p = float(p)
        self.gamma, self.rollout_limit, self.simulation_limit = gamma, rollout_limit, simulation_limit
        self.reward_offset, self.reward_scale, self.init_q = internal_reward_offset, internal_reward_scale, init_q
        
        # Enforce procedural dynamics
        try:
            self.step_fn, self.rollout_fn, self.params = env.get_procedural_dynamics()
        except (AttributeError, NotImplementedError) as e:
            raise RuntimeError(f"GSPUCTFull requires procedural dynamics: {e}")
            
        self.horizon = int(math.ceil(math.log(self.simulation_limit) / (2 * math.log(1.0 / self.gamma))))

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        n_actions = self.env.action_space_size
        max_n = self.simulation_limit * 10 + 1
        T_s, Q_hat, V_hat = np.zeros(max_n, dtype=np.int32), np.full((max_n, n_actions), self.init_q, dtype=np.float32), np.full(max_n, self.init_q, dtype=np.float32)
        T_sa, n_states, n_counter = np.zeros((max_n, n_actions), dtype=np.int32), np.zeros(max_n, dtype=np.int32), np.array([1], dtype=np.int32)
        
        n_states[0] = initial_state
        state_to_node = NumbaDict.empty(key_type=numba.types.int32, value_type=numba.types.int32)
        state_to_node[np.int32(initial_state)] = np.int32(0)
        
        gsp_uct_full_core(self.simulation_limit, self.horizon, initial_state, self.c, self.gamma, self.p, self.rollout_limit, 
                         self.step_fn, self.rollout_fn, self.params, n_actions,
                         T_s, Q_hat, V_hat, T_sa, state_to_node, n_states, n_counter, self.reward_offset, self.reward_scale)
        
        return int(np.argmax(Q_hat[0])), {"root_v": float(V_hat[0]), "horizon": self.horizon, "nodes": int(n_counter[0])}

    def get_name(self) -> str:
        return f"gsp_uct_f_p{self.p}_sim{self.simulation_limit}"
