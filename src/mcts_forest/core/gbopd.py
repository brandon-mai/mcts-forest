import numpy as np
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
from mcts_forest.core.base import sample_discrete_transition

@njit(cache=True)
def gbopd_core(initial_state, transitions, rewards, dones, probs_cum, 
               gamma, simulation_limit, max_n, 
               v_min, v_max, tol,
               v_lower, v_upper, state_to_node, node_to_state, 
               curr_max_n, n_obs, r_sum, d_count,
               successor_nodes, parent_nodes, n_parents,
               reward_scale, reward_offset):
    
    n_actions = successor_nodes.shape[1]
    queue = np.empty(max_n, dtype=np.int32)
    in_queue = np.zeros(max_n, dtype=np.bool_)
    
    for n_iter in range(simulation_limit):
        # 1. Optimistic Sampling
        curr_s = initial_state
        curr_node = np.int32(0)
        
        # Prevent infinite loops in cycles
        sampling_timeout = 100 
        d_cnt = 0
        
        while d_cnt < sampling_timeout:
            d_cnt += 1
            # Check if leaf
            is_leaf = False
            for a in range(n_actions):
                if successor_nodes[curr_node, a] == -1:
                    is_leaf = True
                    break
            
            if is_leaf:
                break
                
            # Optimistic selection
            best_a = -1
            max_val = -1e18
            for a in range(n_actions):
                nxt_node = successor_nodes[curr_node, a]
                if n_obs[curr_node, a] == 0:
                    val = 1e18 # Unknown
                else:
                    r = r_sum[curr_node, a] / n_obs[curr_node, a]
                    if d_count[curr_node, a] > 0:
                        val = r
                    else:
                        val = r + gamma * v_upper[nxt_node]
                    
                if val > max_val:
                    max_val = val
                    best_a = a
            
            # Traversal
            s_nxt, r, d = sample_discrete_transition(curr_s, best_a, transitions, rewards, dones, probs_cum)
            s_nxt_i = np.int32(s_nxt)
            
            # If the next state is unknown in our graph, we stop traversal and expand
            if s_nxt_i not in state_to_node:
                break
                
            nxt_node_id = state_to_node[s_nxt_i]
            # If this action hasn't been linked to this node in the graph yet, we stop and expand
            if successor_nodes[curr_node, best_a] == -1:
                break
                
            curr_s = s_nxt
            curr_node = nxt_node_id
            if d: break
        else:
            # Reached depth limit or loop
            continue
            
        # 2. Node Expansion (on the leaf node)
        if successor_nodes[curr_node, 0] == -1:
            expanded_node = curr_node
            for a in range(n_actions):
                s_nxt, r, d_nxt = sample_discrete_transition(curr_s, a, transitions, rewards, dones, probs_cum)
                s_nxt_i = np.int32(s_nxt)
                
                # Update stats with internal scaling
                scaled_r = (r + reward_offset) * reward_scale
                n_obs[expanded_node, a] += 1
                r_sum[expanded_node, a] += scaled_r
                if d_nxt: d_count[expanded_node, a] += 1
                
                if s_nxt_i not in state_to_node:
                    if curr_max_n[0] < max_n:
                        new_node = np.int32(curr_max_n[0])
                        state_to_node[s_nxt_i] = new_node
                        node_to_state[new_node] = s_nxt_i
                        v_lower[new_node], v_upper[new_node] = v_min, v_max
                        curr_max_n[0] += 1
                        nxt_node_id = new_node
                    else: nxt_node_id = -1
                else:
                    nxt_node_id = state_to_node[s_nxt_i]
                
                if nxt_node_id != -1:
                    successor_nodes[expanded_node, a] = nxt_node_id
                    found_parent = False
                    for p_idx in range(n_parents[nxt_node_id]):
                        if parent_nodes[nxt_node_id, p_idx] == expanded_node:
                            found_parent = True
                            break
                    if not found_parent and n_parents[nxt_node_id] < parent_nodes.shape[1]:
                        parent_nodes[nxt_node_id, n_parents[nxt_node_id]] = expanded_node
                        n_parents[nxt_node_id] += 1

            # 3. Partial Value Iteration
            head, tail = 0, 0
            queue[tail] = expanded_node
            tail += 1
            in_queue[expanded_node] = True
            
            sweeps = 0
            while head < tail and sweeps < 1000:
                s_prime = queue[head]
                head += 1
                in_queue[s_prime] = False
                sweeps += 1
                
                old_l, old_u = v_lower[s_prime], v_upper[s_prime]
                best_l, best_u = -1e18, -1e18
                
                for a in range(n_actions):
                    if n_obs[s_prime, a] == 0: continue
                    nxt = successor_nodes[s_prime, a]
                    r = r_sum[s_prime, a] / n_obs[s_prime, a]
                    if nxt == -1:
                        val_l, val_u = v_min, v_max
                    elif d_count[s_prime, a] > 0:
                        val_l, val_u = r, r
                    else:
                        val_l = r + gamma * v_lower[nxt]
                        val_u = r + gamma * v_upper[nxt]
                    
                    if val_l > best_l: best_l = val_l
                    if val_u > best_u: best_u = val_u
                
                v_lower[s_prime], v_upper[s_prime] = best_l, best_u
                
                if max(abs(old_l - best_l), abs(old_u - best_u)) > tol:
                    for p_idx in range(n_parents[s_prime]):
                        p_node = parent_nodes[s_prime, p_idx]
                        if not in_queue[p_node]:
                            queue[tail] = p_node
                            tail += 1
                            in_queue[p_node] = True
                            
            for i in range(tail):
                in_queue[queue[i]] = False

    return curr_max_n[0]

class GBOPD:
    def __init__(self, env, gamma=0.99, simulation_limit=100, v_min=0.0, v_max=1.0, tol=1e-3, **kwargs):
        self.env, self.gamma, self.simulation_limit = env, gamma, simulation_limit
        self.v_min, self.v_max, self.tol = v_min, v_max, tol
        self.reward_scale = kwargs.get("internal_reward_scale", 1.0)
        self.reward_offset = kwargs.get("internal_reward_offset", 0.0)
        self.dynamics = env.get_numba_dynamics()
        self.max_n = 5000
        self.max_parents = 64

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        n_actions = self.dynamics[0].shape[1]
        v_lower = np.full(self.max_n, self.v_min, dtype=np.float32)
        v_upper = np.full(self.max_n, self.v_max, dtype=np.float32)
        state_to_node = NumbaDict.empty(numba.int32, numba.int32)
        node_to_state = np.zeros(self.max_n, dtype=np.int32)
        successor_nodes = np.full((self.max_n, n_actions), -1, dtype=np.int32)
        n_obs = np.zeros((self.max_n, n_actions), dtype=np.int32)
        r_sum = np.zeros((self.max_n, n_actions), dtype=np.float32)
        d_count = np.zeros((self.max_n, n_actions), dtype=np.int32)
        
        parent_nodes = np.full((self.max_n, self.max_parents), -1, dtype=np.int32)
        n_parents = np.zeros(self.max_n, dtype=np.int32)
        
        state_to_node[np.int32(initial_state)], node_to_state[0] = np.int32(0), np.int32(initial_state)
        curr_max_n = np.array([1], dtype=np.int32)
        
        final_max_n = gbopd_core(
            initial_state, *self.dynamics, self.gamma, self.simulation_limit, 
            self.max_n, self.v_min, self.v_max, self.tol,
            v_lower, v_upper, state_to_node, node_to_state, curr_max_n,
            n_obs, r_sum, d_count, successor_nodes,
            parent_nodes, n_parents,
            self.reward_scale, self.reward_offset
        )
        
        best_a, max_l = -1, -1e18
        for a in range(n_actions):
            if n_obs[0, a] == 0:
                val = self.v_min
            else:
                r = r_sum[0, a] / n_obs[0, a]
                nxt = successor_nodes[0, a]
                if d_count[0, a] > 0:
                    val = r
                elif nxt != -1:
                    val = r + self.gamma * v_lower[nxt]
                else:
                    val = r + self.gamma * self.v_min
                    
            if val > max_l:
                max_l, best_a = val, a
                
        return best_a, {"root_v": float(v_lower[0]), "nodes": int(final_max_n)}

    def get_name(self) -> str:
        return f"gbopd_sim{self.simulation_limit}"

    def get_name(self) -> str:
        return f"gbopd_sim{self.simulation_limit}"
