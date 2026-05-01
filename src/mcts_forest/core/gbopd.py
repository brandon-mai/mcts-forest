import numpy as np
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
from mcts_forest.core.base import sample_discrete_transition

@njit
def gbopd_core(initial_state, transitions, rewards, dones, probs_cum, 
               gamma, simulation_limit, max_n, 
               v_min, v_max, tol, max_sweeps,
               v_lower, v_upper, state_to_node, node_to_state, 
               curr_max_n, child_nodes, child_rewards, child_dones):
    
    n_actions = child_nodes.shape[1]
    
    for _ in range(simulation_limit):
        # 1. Compute Global Exact Bounds (Value Iteration on the current graph)
        for sweep in range(max_sweeps if max_sweeps > 0 else 1000):
            max_diff = 0.0
            for n_id in range(curr_max_n):
                # Update Lower Bound
                old_l = v_lower[n_id]
                best_l = -1e18
                # Update Upper Bound
                old_u = v_upper[n_id]
                best_u = -1e18
                
                for a in range(n_actions):
                    nxt_node = child_nodes[n_id, a]
                    r = child_rewards[n_id, a]
                    is_done = child_dones[n_id, a]
                    
                    if nxt_node == -1: # Unexpanded
                        val_l = v_min
                        val_u = v_max
                    else:
                        if is_done:
                            val_l = r
                            val_u = r
                        else:
                            val_l = r + gamma * v_lower[nxt_node]
                            val_u = r + gamma * v_upper[nxt_node]
                    
                    if val_l > best_l: best_l = val_l
                    if val_u > best_u: best_u = val_u
                
                v_lower[n_id] = best_l
                v_upper[n_id] = best_u
                
                diff = max(abs(old_l - best_l), abs(old_u - best_u))
                if diff > max_diff: max_diff = diff
            
            if max_sweeps <= 0 and max_diff < tol:
                break

        # 2. Optimistic Sampling (Greedy on Upper Bound)
        curr_s = initial_state
        curr_node = np.int32(0)
        
        # Traverse until we hit an unexpanded node or a terminal state
        while True:
            # Check if current node has any unexpanded actions
            has_unexpanded = False
            for a in range(n_actions):
                if child_nodes[curr_node, a] == -1:
                    has_unexpanded = True
                    break
            
            if has_unexpanded:
                break
                
            # Select action maximizing r + gamma * U(s')
            best_a = -1
            max_val = -1e18
            for a in range(n_actions):
                nxt_node = child_nodes[curr_node, a]
                r = child_rewards[curr_node, a]
                is_done = child_dones[curr_node, a]
                
                if is_done:
                    val = r
                else:
                    val = r + gamma * v_upper[nxt_node]
                    
                if val > max_val:
                    max_val = val
                    best_a = a
            
            s_nxt, r, d = sample_discrete_transition(curr_s, best_a, transitions, rewards, dones, probs_cum)
            curr_s = s_nxt
            curr_node = child_nodes[curr_node, best_a]
            if d: break
            
        # 3. Node Expansion (Expand all actions at the leaf)
        if not d:
            for a in range(n_actions):
                if child_nodes[curr_node, a] == -1:
                    s_nxt, r, d_nxt = sample_discrete_transition(curr_s, a, transitions, rewards, dones, probs_cum)
                    s_nxt_i = np.int32(s_nxt)
                    
                    child_rewards[curr_node, a] = r
                    child_dones[curr_node, a] = d_nxt
                    
                    if s_nxt_i not in state_to_node:
                        if curr_max_n < max_n:
                            new_node = np.int32(curr_max_n)
                            state_to_node[s_nxt_i] = new_node
                            node_to_state[new_node] = s_nxt_i
                            # Initialize new node bounds
                            v_lower[new_node] = v_min
                            v_upper[new_node] = v_max
                            curr_max_n += 1
                            child_nodes[curr_node, a] = new_node
                    else:
                        child_nodes[curr_node, a] = state_to_node[s_nxt_i]
    
    return curr_max_n

class GBOPD:
    def __init__(self, env, gamma=0.99, simulation_limit=100, v_min=-1000.0, v_max=2000.0, tol=1e-4, max_sweeps=0, **kwargs):
        self.env = env
        self.gamma = gamma
        self.simulation_limit = simulation_limit
        self.v_min = v_min
        self.v_max = v_max
        self.tol = tol
        self.max_sweeps = max_sweeps
        self.dynamics = env.get_numba_dynamics()
        self.max_n = 10000

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        max_n = self.max_n
        n_actions = self.dynamics[0].shape[1]
        
        v_lower = np.full(max_n, self.v_min, dtype=np.float32)
        v_upper = np.full(max_n, self.v_max, dtype=np.float32)
        
        state_to_node = NumbaDict.empty(numba.int32, numba.int32)
        node_to_state = np.zeros(max_n, dtype=np.int32)
        
        child_nodes = np.full((max_n, n_actions), -1, dtype=np.int32)
        child_rewards = np.zeros((max_n, n_actions), dtype=np.float32)
        child_dones = np.zeros((max_n, n_actions), dtype=np.bool_)
        
        state_to_node[np.int32(initial_state)] = np.int32(0)
        node_to_state[0] = np.int32(initial_state)
        
        final_max_n = gbopd_core(
            initial_state, *self.dynamics, self.gamma, self.simulation_limit, 
            max_n, self.v_min, self.v_max, self.tol, self.max_sweeps,
            v_lower, v_upper, state_to_node, node_to_state, 1, 
            child_nodes, child_rewards, child_dones
        )
        
        # Recommendation Rule: Conservative (Lower Bound)
        best_a = -1
        max_val = -1e18
        for a in range(n_actions):
            nxt_node = child_nodes[0, a]
            r = child_rewards[0, a]
            is_done = child_dones[0, a]
            
            if is_done:
                val = r
            elif nxt_node != -1:
                val = r + self.gamma * v_lower[nxt_node]
            else:
                val = r + self.gamma * self.v_min
                
            if val > max_val:
                max_val = val
                best_a = a
        
        return best_a, {"root_v": float(v_lower[0]), "nodes": int(final_max_n)}

    def get_name(self) -> str:
        return f"gbopd_sim{self.simulation_limit}"
