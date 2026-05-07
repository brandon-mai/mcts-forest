import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any

@njit
def mcgs_core(initial_state, step_fn, rollout_fn, params, n_actions, c, horizon, gamma, rollout_limit, simulation_limit, max_n, visit_count, q_hat, action_visits, state_to_node, node_to_state, curr_max_n_arr, child_nodes, reward_offset, reward_scale):
    p_nodes = np.zeros(horizon + 1, dtype=np.int32)
    p_a = np.zeros(horizon + 1, dtype=np.int32)
    p_r_int = np.zeros(horizon + 1, dtype=np.float32)
    
    for _ in range(simulation_limit):
        curr_s = initial_state
        curr_node = np.int32(0)
        p_len = 0
        v_leaf = 0.0
        
        while p_len < horizon:
            best_a = -1
            max_u = -1e18
            n_p = visit_count[curr_node]
            ln_p = math.log(float(max(1, n_p)))
            # Selection
            found_unvisited = False
            for a in range(n_actions):
                if action_visits[curr_node, a] == 0:
                    best_a = a
                    found_unvisited = True
                    break
            
            if not found_unvisited:
                for a in range(n_actions):
                    n_sa = action_visits[curr_node, a]
                    val = q_hat[curr_node, a] + c * math.sqrt(ln_p / n_sa)
                    if val > max_u:
                        max_u = val
                        best_a = a
            
            # Step
            s_nxt, r, d = step_fn(curr_s, best_a, *params)
            
            # Internal normalization
            r_int = (r + reward_offset) * reward_scale
            
            p_nodes[p_len] = curr_node
            p_a[p_len] = best_a
            p_r_int[p_len] = r_int
            p_len += 1
            
            s_nxt_i = np.int32(s_nxt)
            if s_nxt_i not in state_to_node:
                if curr_max_n_arr[0] < max_n:
                    new_node = np.int32(curr_max_n_arr[0])
                    state_to_node[s_nxt_i] = new_node
                    node_to_state[new_node] = s_nxt_i
                    curr_max_n_arr[0] += 1
                    child_nodes[curr_node, best_a] = new_node
                    # Procedural Rollout
                    v_leaf = (rollout_fn(s_nxt, *params, rollout_limit, gamma) + reward_offset) * reward_scale
                    break
                else:
                    v_leaf = (rollout_fn(s_nxt, *params, rollout_limit, gamma) + reward_offset) * reward_scale
                    break
            else:
                next_node = state_to_node[s_nxt_i]
                child_nodes[curr_node, best_a] = next_node
                curr_node = next_node
                curr_s = s_nxt
                if d:
                    v_leaf = 0.0
                    break
        else:
            v_leaf = (rollout_fn(curr_s, *params, rollout_limit, gamma) + reward_offset) * reward_scale

        # Backpropagation
        v_nxt = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r_int = p_nodes[i], p_a[i], p_r_int[i]
            action_visits[n_id, a] += 1
            visit_count[n_id] += 1
            q_hat[n_id, a] += (r_int + gamma * v_nxt - q_hat[n_id, a]) / action_visits[n_id, a]
            # Bootstrap
            v_nxt = float(-1e9)
            for aa in range(n_actions):
                if q_hat[n_id, aa] > v_nxt:
                    v_nxt = q_hat[n_id, aa]
    
    return curr_max_n_arr[0]

class MCGS:
    def __init__(self, env, c=1.0, horizon=100, gamma=0.99, rollout_limit=100, simulation_limit=1000, 
                 internal_reward_offset=0.0, internal_reward_scale=1.0, init_q=0.0, **kwargs):
        self.env = env
        self.c, self.horizon, self.gamma, self.rollout_limit, self.simulation_limit = c, horizon, gamma, rollout_limit, simulation_limit
        self.reward_offset, self.reward_scale, self.init_q = internal_reward_offset, internal_reward_scale, init_q
        
        # Enforce procedural dynamics
        try:
            self.step_fn, self.rollout_fn, self.params = env.get_procedural_dynamics()
        except (AttributeError, NotImplementedError) as e:
            raise RuntimeError(f"MCGS requires procedural dynamics: {e}")
            
        self.max_n = 20000

    def search(self, initial_state):
        n_actions = self.env.action_space_size
        max_n = self.max_n
        v_count = np.zeros(max_n, dtype=np.int32)
        q_hat = np.full((max_n, n_actions), self.init_q, dtype=np.float32)
        a_visits = np.zeros((max_n, n_actions), dtype=np.int32)
        state_to_node = NumbaDict.empty(numba.int32, numba.int32)
        node_to_state = np.zeros(max_n, dtype=np.int32)
        child_nodes = np.full((max_n, n_actions), -1, dtype=np.int32)
        
        state_to_node[np.int32(initial_state)] = np.int32(0)
        node_to_state[0] = np.int32(initial_state)
        curr_max_n_arr = np.array([1], dtype=np.int32)
        
        final_max_n = mcgs_core(
            initial_state, self.step_fn, self.rollout_fn, self.params, n_actions, self.c, self.horizon, self.gamma, self.rollout_limit, self.simulation_limit, 
            max_n, v_count, q_hat, a_visits, state_to_node, node_to_state, curr_max_n_arr, child_nodes, self.reward_offset, self.reward_scale
        )
        
        best_a = int(np.argmax(q_hat[0]))
        return best_a, {"root": None, "root_v": float(np.max(q_hat[0]))}

    def get_name(self):
        return f"mcgs_sim{self.simulation_limit}"
