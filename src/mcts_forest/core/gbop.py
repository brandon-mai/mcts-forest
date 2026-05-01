import numpy as np
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
import math
from mcts_forest.core.base import sample_discrete_transition

@njit
def kl_upper_bound(p_hat, v, epsilon, v_max):
    if epsilon <= 0 or len(p_hat) <= 1:
        return np.sum(p_hat * v)
    
    v_max_val = -1e18
    for i in range(len(v)):
        if v[i] > v_max_val: v_max_val = v[i]
    
    # Simple binary search for the dual variable lambda
    low = v_max_val + 1e-7
    high = v_max_val + 1e6
    
    for _ in range(20):
        mid = (low + high) / 2.0
        # Calculate KL and current expectation
        sum_p = 0.0
        for i in range(len(v)):
            sum_p += p_hat[i] / (mid - v[i])
        
        kl = math.log(sum_p)
        for i in range(len(v)):
            if p_hat[i] > 0:
                kl += p_hat[i] * math.log(mid - v[i])
        
        if kl > epsilon: low = mid
        else: high = mid
            
    sum_p = 0.0
    for i in range(len(v)):
        sum_p += p_hat[i] / (high - v[i])
    
    return high - 1.0/sum_p

@njit
def kl_lower_bound(p_hat, v, epsilon, v_min):
    # min sum p_i v_i = - max sum p_i (-v_i)
    return -kl_upper_bound(p_hat, -v, epsilon, -v_min)

@njit
def gbop_core(initial_state, transitions, rewards, dones, probs_cum, 
              gamma, horizon, m_trajectories, max_n, 
              v_min, v_max, tol, max_sweeps,
              v_lower, v_upper, state_to_node, node_to_state, 
              curr_max_n, n_obs, r_sum, d_count, 
              n_successors, successor_nodes, successor_counts):
    
    n_actions = n_obs.shape[1]
    n_total = 0
    
    for m in range(m_trajectories):
        curr_s = initial_state
        curr_node = np.int32(0)
        
        for t in range(horizon):
            n_total += 1
            
            # 1. Compute Global Stochastic Bounds (Iterative Bellman)
            # We only run VI if we've seen enough samples to change the bounds
            # For efficiency, we can limit sweeps
            sweeps = max_sweeps if max_sweeps > 0 else 20
            for sweep in range(sweeps):
                max_diff = 0.0
                for n_id in range(curr_max_n):
                    old_l = v_lower[n_id]
                    old_u = v_upper[n_id]
                    
                    best_l = -1e18
                    best_u = -1e18
                    
                    for a in range(n_actions):
                        n_sa = n_obs[n_id, a]
                        if n_sa == 0:
                            ql, qu = v_min, v_max
                        else:
                            r_mean = r_sum[n_id, a] / n_sa
                            p_done = d_count[n_id, a] / n_sa
                            epsilon = math.log(n_total) / n_sa
                            
                            k_max = n_successors[n_id, a]
                            if k_max == 0: # Should not happen if n_sa > 0 unless all terminal
                                ql, qu = r_mean, r_mean
                            else:
                                p_hat = np.zeros(k_max, dtype=np.float32)
                                v_next_l = np.zeros(k_max, dtype=np.float32)
                                v_next_u = np.zeros(k_max, dtype=np.float32)
                                for k in range(k_max):
                                    nxt_id = successor_nodes[n_id, a, k]
                                    p_hat[k] = successor_counts[n_id, a, k] / n_sa
                                    v_next_l[k] = v_lower[nxt_id]
                                    v_next_u[k] = v_upper[nxt_id]
                                
                                ql = r_mean + gamma * (1.0 - p_done) * kl_lower_bound(p_hat, v_next_l, epsilon, v_min)
                                qu = r_mean + gamma * (1.0 - p_done) * kl_upper_bound(p_hat, v_next_u, epsilon, v_max)
                        
                        if ql > best_l: best_l = ql
                        if qu > best_u: best_u = qu
                    
                    v_lower[n_id] = best_l
                    v_upper[n_id] = best_u
                    max_diff = max(max_diff, max(abs(old_l-best_l), abs(old_u-best_u)))
                
                if max_sweeps <= 0 and max_diff < tol: break
                
            # 2. Optimistic Selection
            best_a = 0
            max_q = -1e18
            for a in range(n_actions):
                n_sa = n_obs[curr_node, a]
                if n_sa == 0:
                    qu = v_max
                else:
                    # Recalculate qu for the current node precisely
                    r_mean = r_sum[curr_node, a] / n_sa
                    p_done = d_count[curr_node, a] / n_sa
                    epsilon = math.log(n_total) / n_sa
                    k_max = n_successors[curr_node, a]
                    if k_max == 0: qu = r_mean
                    else:
                        p_hat = np.zeros(k_max, dtype=np.float32)
                        v_next_u = np.zeros(k_max, dtype=np.float32)
                        for k in range(k_max):
                            nxt_id = successor_nodes[curr_node, a, k]
                            p_hat[k] = successor_counts[curr_node, a, k] / n_sa
                            v_next_u[k] = v_upper[nxt_id]
                        qu = r_mean + gamma * (1.0 - p_done) * kl_upper_bound(p_hat, v_next_u, epsilon, v_max)
                
                if qu > max_q:
                    max_q = qu
                    best_a = a
            
            # 3. Step
            s_nxt, r, d = sample_discrete_transition(curr_s, best_a, transitions, rewards, dones, probs_cum)
            s_nxt_i = np.int32(s_nxt)
            
            # 4. Record
            n_obs[curr_node, best_a] += 1
            r_sum[curr_node, best_a] += r
            if d: d_count[curr_node, best_a] += 1
            
            if s_nxt_i not in state_to_node:
                if curr_max_n < max_n:
                    new_id = np.int32(curr_max_n)
                    state_to_node[s_nxt_i] = new_id
                    node_to_state[new_id] = s_nxt_i
                    v_lower[new_id], v_upper[new_id] = v_min, v_max
                    curr_max_n += 1
                    nxt_node_id = new_id
                else: nxt_node_id = -1
            else: nxt_node_id = state_to_node[s_nxt_i]
            
            if nxt_node_id != -1 and not d:
                # Update successor list
                found = False
                for k in range(n_successors[curr_node, best_a]):
                    if successor_nodes[curr_node, best_a, k] == nxt_node_id:
                        successor_counts[curr_node, best_a, k] += 1
                        found = True
                        break
                if not found:
                    k = n_successors[curr_node, best_a]
                    if k < successor_nodes.shape[2]:
                        successor_nodes[curr_node, best_a, k] = nxt_node_id
                        successor_counts[curr_node, best_a, k] = 1
                        n_successors[curr_node, best_a] += 1
                
                curr_node = nxt_node_id
                curr_s = s_nxt
            
            if d: break
            
    return curr_max_n

class GBOP:
    def __init__(self, env, gamma=0.99, horizon=100, trajectories=100, v_min=-1000.0, v_max=2000.0, tol=1e-4, max_sweeps=0, **kwargs):
        self.env = env
        self.gamma = gamma
        self.horizon = horizon
        self.trajectories = trajectories
        self.v_min = v_min
        self.v_max = v_max
        self.tol = tol
        self.max_sweeps = max_sweeps
        self.dynamics = env.get_numba_dynamics()
        self.max_n = 2000
        self.max_k = 10 # Max successors per action

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        n_actions = self.dynamics[0].shape[1]
        v_lower = np.full(self.max_n, self.v_min, dtype=np.float32)
        v_upper = np.full(self.max_n, self.v_max, dtype=np.float32)
        state_to_node = NumbaDict.empty(numba.int32, numba.int32)
        node_to_state = np.zeros(self.max_n, dtype=np.int32)
        n_obs = np.zeros((self.max_n, n_actions), dtype=np.int32)
        r_sum = np.zeros((self.max_n, n_actions), dtype=np.float32)
        d_count = np.zeros((self.max_n, n_actions), dtype=np.int32)
        n_successors = np.zeros((self.max_n, n_actions), dtype=np.int32)
        successor_nodes = np.full((self.max_n, n_actions, self.max_k), -1, dtype=np.int32)
        successor_counts = np.zeros((self.max_n, n_actions, self.max_k), dtype=np.int32)
        
        state_to_node[np.int32(initial_state)] = np.int32(0)
        node_to_state[0] = np.int32(initial_state)
        
        final_max_n = gbop_core(
            initial_state, *self.dynamics, self.gamma, self.horizon, self.trajectories,
            self.max_n, self.v_min, self.v_max, self.tol, self.max_sweeps,
            v_lower, v_upper, state_to_node, node_to_state, 1,
            n_obs, r_sum, d_count, n_successors, successor_nodes, successor_counts
        )
        
        # Correct recommendation: argmax over conservative Q-values
        best_a = 0
        max_ql = -1e18
        for a in range(n_actions):
            n_sa = n_obs[0, a]
            if n_sa == 0: ql = self.v_min
            else:
                r_mean = r_sum[0, a] / n_sa
                p_done = d_count[0, a] / n_sa
                epsilon = math.log(self.trajectories * self.horizon) / n_sa
                k_max = n_successors[0, a]
                if k_max == 0: ql = r_mean
                else:
                    p_hat = np.zeros(k_max, dtype=np.float32)
                    v_next_l = np.zeros(k_max, dtype=np.float32)
                    for k in range(k_max):
                        nxt_id = successor_nodes[0, a, k]
                        p_hat[k] = successor_counts[0, a, k] / n_sa
                        v_next_l[k] = v_lower[nxt_id]
                    ql = r_mean + self.gamma * (1.0 - p_done) * kl_lower_bound(p_hat, v_next_l, epsilon, self.v_min)
            
            if ql > max_ql:
                max_ql = ql
                best_a = a
                    
        return best_a, {"root_v": float(v_lower[0]), "nodes": int(final_max_n)}

    def get_name(self) -> str:
        return f"gbop_traj{self.trajectories}"
