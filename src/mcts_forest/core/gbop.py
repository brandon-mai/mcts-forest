import numpy as np
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
import math

@njit
def max_expectation_under_constraint(f, q, c, eps=1e-3, use_newton=True):
    if c <= 0 or len(q) <= 1:
        sum_qf = 0.0
        for i in range(len(q)):
            sum_qf += q[i] * f[i]
        return sum_qf
    f_star = np.amax(f)
    if np.isclose(f, f[0]).all(): return f[0]
    a = f_star + 1e-7
    b = f_star + 100.0
    for _ in range(10):
        sum_inv, sum_log = 0.0, 0.0
        for i in range(len(q)):
            diff = b - f[i]
            sum_inv += q[i] / diff
            if q[i] > 1e-10: sum_log += q[i] * math.log(diff)
        theta_b = sum_log + math.log(max(sum_inv, 1e-12)) - c
        if theta_b <= 0: break
        b = f_star + (b - f_star) * 2.0
    res_l = b
    newton_converged = False
    if use_newton:
        l = f_star + 1.0
        for _ in range(10):
            if l <= f_star: l = (f_star + b) / 2.0
            sum_inv, sum_inv2, sum_log = 0.0, 0.0, 0.0
            for i in range(len(q)):
                diff = l - f[i]
                inv = 1.0 / diff
                sum_inv += q[i] * inv
                sum_inv2 += q[i] * inv * inv
                if q[i] > 1e-10: sum_log += q[i] * math.log(diff)
            theta = sum_log + math.log(max(sum_inv, 1e-12)) - c
            d_theta = sum_inv - sum_inv2 / max(sum_inv, 1e-12)
            if abs(theta) < eps:
                res_l = l
                newton_converged = True
                break
            if abs(d_theta) < 1e-10: break
            l_next = l - theta / d_theta
            if l_next <= f_star or l_next > b: break
            l = l_next
    if not newton_converged:
        for _ in range(30):
            mid = (a + b) / 2.0
            sum_inv, sum_log = 0.0, 0.0
            for i in range(len(q)):
                diff = mid - f[i]
                sum_inv += q[i] / diff
                if q[i] > 1e-10: sum_log += q[i] * math.log(diff)
            theta_mid = sum_log + math.log(max(sum_inv, 1e-12)) - c
            if theta_mid > 0: a = mid
            else: b = mid; res_l = mid
    sum_inv = 0.0
    for i in range(len(q)): sum_inv += q[i] / (res_l - f[i])
    beta = 1.0 / max(sum_inv, 1e-12)
    exp_p = 0.0
    for i in range(len(q)): exp_p += (beta * q[i] / (res_l - f[i])) * f[i]
    return exp_p

@njit
def min_expectation_under_constraint(f, q, c, eps=1e-3, use_newton=True):
    return -max_expectation_under_constraint(-f, q, c, eps, use_newton)

@njit
def gbop_core(initial_state, step_fn, params, n_actions,
              gamma, horizon, trajectories, max_n, 
              v_min, v_max, tol,
              v_lower, v_upper, state_to_node, node_to_state, 
              curr_max_n, n_obs, r_sum, d_count, 
              n_successors, successor_nodes, successor_counts,
              parent_nodes, n_parents,
              p_buf, v_l_buf, v_u_buf, use_newton,
              reward_scale, reward_offset):
    
    n_total = 0
    queue = np.empty(max_n, dtype=np.int32)
    in_queue = np.zeros(max_n, dtype=np.bool_)
    traj_nodes = np.empty(horizon, dtype=np.int32)
    
    for m in range(trajectories):
        curr_s = initial_state
        curr_node = np.int32(0)
        p_len = 0
        for t in range(horizon):
            n_total += 1
            traj_nodes[p_len] = curr_node
            p_len += 1
            best_a = 0
            max_q = -1e18
            for a in range(n_actions):
                n_sa = n_obs[curr_node, a]
                if n_sa == 0: qu = v_max
                else:
                    r_mean, p_done, k_max = r_sum[curr_node, a] / n_sa, d_count[curr_node, a] / n_sa, n_successors[curr_node, a]
                    epsilon = math.log(n_total) / n_sa
                    if k_max == 0: qu = r_mean + gamma * (1.0 - p_done) * v_max
                    elif k_max == 1: qu = r_mean + gamma * (1.0 - p_done) * v_upper[successor_nodes[curr_node, a, 0]]
                    else:
                        p_hat, v_next_u = p_buf[:k_max], v_u_buf[:k_max]
                        for k in range(k_max):
                            nid = successor_nodes[curr_node, a, k]
                            p_hat[k], v_next_u[k] = successor_counts[curr_node, a, k] / n_sa, v_upper[nid]
                        qu = r_mean + gamma * (1.0 - p_done) * max_expectation_under_constraint(v_next_u, p_hat, epsilon, use_newton=use_newton)
                if qu > max_q: max_q, best_a = qu, a
            
            # Step Logic
            s_nxt, r, d = step_fn(curr_s, best_a, *params)
            s_nxt_i = np.int32(s_nxt)
            scaled_r = np.float32((r + reward_offset) * reward_scale)
            n_obs[curr_node, best_a] += 1
            r_sum[curr_node, best_a] += scaled_r
            if d: d_count[curr_node, best_a] += 1
            
            if s_nxt_i not in state_to_node:
                if curr_max_n[0] < max_n:
                    new_id = np.int32(curr_max_n[0])
                    state_to_node[s_nxt_i], node_to_state[new_id] = new_id, s_nxt_i
                    v_lower[new_id], v_upper[new_id] = v_min, v_max
                    curr_max_n[0] += 1
                    nxt_node_id = new_id
                else: nxt_node_id = -1
            else: nxt_node_id = state_to_node[s_nxt_i]
            
            if nxt_node_id != -1:
                found_succ = False
                for k in range(n_successors[curr_node, best_a]):
                    if successor_nodes[curr_node, best_a, k] == nxt_node_id:
                        successor_counts[curr_node, best_a, k] += 1
                        found_succ = True; break
                if not found_succ:
                    k = n_successors[curr_node, best_a]
                    if k < successor_nodes.shape[2]:
                        successor_nodes[curr_node, best_a, k], successor_counts[curr_node, best_a, k] = nxt_node_id, 1
                        n_successors[curr_node, best_a] += 1
                found_parent = False
                for p_idx in range(n_parents[nxt_node_id]):
                    if parent_nodes[nxt_node_id, p_idx] == curr_node:
                        found_parent = True; break
                if not found_parent and n_parents[nxt_node_id] < parent_nodes.shape[1]:
                    parent_nodes[nxt_node_id, n_parents[nxt_node_id]] = curr_node
                    n_parents[nxt_node_id] += 1
                curr_node, curr_s = nxt_node_id, s_nxt
            else: break
            if d: break
            
        head, tail = 0, 0
        for i in range(p_len - 1, -1, -1):
            s_traj = traj_nodes[i]
            if not in_queue[s_traj]: queue[tail] = s_traj; tail += 1; in_queue[s_traj] = True
        sweeps = 0
        while head < tail and sweeps < 1000:
            s_prime = queue[head]; head += 1; in_queue[s_prime] = False; sweeps += 1
            old_l, old_u = v_lower[s_prime], v_upper[s_prime]
            best_l, best_u = -1e18, -1e18
            for a in range(n_actions):
                n_sa = n_obs[s_prime, a]
                if n_sa == 0: ql, qu = v_min, v_max
                else:
                    r_mean, p_done, k_max = r_sum[s_prime, a] / n_sa, d_count[s_prime, a] / n_sa, n_successors[s_prime, a]
                    epsilon = math.log(n_total) / n_sa
                    if k_max == 0: ql, qu = r_mean + gamma*(1.0-p_done)*v_min, r_mean + gamma*(1.0-p_done)*v_max
                    elif k_max == 1:
                        nid = successor_nodes[s_prime, a, 0]
                        ql, qu = r_mean + gamma*(1.0-p_done)*v_lower[nid], r_mean + gamma*(1.0-p_done)*v_upper[nid]
                    else:
                        p_hat, v_next_l, v_next_u = p_buf[:k_max], v_l_buf[:k_max], v_u_buf[:k_max]
                        for k in range(k_max):
                            nid = successor_nodes[s_prime, a, k]
                            p_hat[k], v_next_l[k], v_next_u[k] = successor_counts[s_prime, a, k] / n_sa, v_lower[nid], v_upper[nid]
                        ql = r_mean + gamma * (1.0 - p_done) * min_expectation_under_constraint(v_next_l, p_hat, epsilon, use_newton=use_newton)
                        qu = r_mean + gamma * (1.0 - p_done) * max_expectation_under_constraint(v_next_u, p_hat, epsilon, use_newton=use_newton)
                if ql > best_l: best_l = ql
                if qu > best_u: best_u = qu
            v_lower[s_prime], v_upper[s_prime] = best_l, best_u
            if max(abs(old_l - best_l), abs(old_u - best_u)) > tol:
                for p_idx in range(n_parents[s_prime]):
                    p_node = parent_nodes[s_prime, p_idx]
                    if not in_queue[p_node]: queue[tail] = p_node; tail += 1; in_queue[p_node] = True
        for i in range(tail): in_queue[queue[i]] = False
    return curr_max_n[0]

class GBOP:
    def __init__(self, env, gamma=0.99, horizon=100, trajectories=100, v_min=0.0, v_max=1.0, tol=1e-3, use_newton=True, **kwargs):
        self.env, self.gamma = env, gamma
        self.v_min, self.v_max, self.tol = v_min, v_max, tol
        self.use_newton = use_newton
        self.reward_scale = np.float32(kwargs.get("internal_reward_scale", 1.0))
        self.reward_offset = np.float32(kwargs.get("internal_reward_offset", 0.0))
        
        # Enforce procedural dynamics
        try:
            self.step_fn, _, self.params = env.get_procedural_dynamics()
        except (AttributeError, NotImplementedError) as e:
            raise RuntimeError(f"GBOP requires procedural dynamics: {e}")

        strategy = kwargs.get("budget_strategy", "auto")
        budget = kwargs.get("simulation_limit", horizon * trajectories)
        if strategy == "manual":
            self.horizon = kwargs.get("horizon", horizon)
            self.trajectories = max(1, budget // self.horizon)
        elif strategy == "generous":
            self.trajectories, self.horizon = budget, kwargs.get("horizon", horizon)
        elif "simulation_limit" in kwargs:
            log_inv_gamma = math.log(1.0 / gamma) if gamma < 1.0 else math.log(1.0 / 0.99)
            best_m, best_l = 1, budget
            low, high = 1, budget
            while low <= high:
                mid = (low + high) // 2
                l_mid = budget if mid <= 1 else max(1, int(math.ceil(math.log(mid) / (2 * log_inv_gamma))))
                if mid * l_mid <= budget: best_m, best_l = mid, l_mid; low = mid + 1
                else: high = mid - 1
            self.trajectories, self.horizon = best_m, best_l
        else: self.horizon, self.trajectories = horizon, trajectories
        self.max_n = max(10000, self.trajectories * self.horizon + 1000) 
        self.max_k, self.max_parents = 64, 64
        self.strategy = strategy

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        n_actions = self.env.action_space_size
        v_lower, v_upper = np.full(self.max_n, self.v_min, dtype=np.float32), np.full(self.max_n, self.v_max, dtype=np.float32)
        state_to_node = NumbaDict.empty(numba.int32, numba.int32)
        node_to_state = np.zeros(self.max_n, dtype=np.int32)
        n_obs, r_sum, d_count = np.zeros((self.max_n, n_actions), dtype=np.int32), np.zeros((self.max_n, n_actions), dtype=np.float32), np.zeros((self.max_n, n_actions), dtype=np.int32)
        n_successors = np.zeros((self.max_n, n_actions), dtype=np.int32)
        successor_nodes = np.full((self.max_n, n_actions, self.max_k), -1, dtype=np.int32)
        successor_counts = np.zeros((self.max_n, n_actions, self.max_k), dtype=np.int32)
        parent_nodes, n_parents = np.full((self.max_n, self.max_parents), -1, dtype=np.int32), np.zeros(self.max_n, dtype=np.int32)
        p_buf, v_l_buf, v_u_buf = np.zeros(self.max_k, dtype=np.float32), np.zeros(self.max_k, dtype=np.float32), np.zeros(self.max_k, dtype=np.float32)
        state_to_node[np.int32(initial_state)], node_to_state[0] = np.int32(0), np.int32(initial_state)
        curr_max_n = np.array([1], dtype=np.int32)
        
        final_max_n = gbop_core(
            initial_state, self.step_fn, self.params, n_actions, self.gamma, self.horizon, self.trajectories,
            self.max_n, self.v_min, self.v_max, self.tol,
            v_lower, v_upper, state_to_node, node_to_state, curr_max_n,
            n_obs, r_sum, d_count, n_successors, successor_nodes, successor_counts,
            parent_nodes, n_parents, p_buf, v_l_buf, v_u_buf, self.use_newton,
            np.float32(self.reward_scale), np.float32(self.reward_offset)
        )
        best_a, max_ql = 0, -1e18
        n_total_root = np.sum(n_obs[0])
        for a in range(n_actions):
            n_sa = n_obs[0, a]
            if n_sa == 0: ql = self.v_min
            else:
                r_mean, p_done, k_max = r_sum[0, a] / n_sa, d_count[0, a] / n_sa, n_successors[0, a]
                if k_max == 0: ql = r_mean + self.gamma * (1.0 - p_done) * self.v_min
                elif k_max == 1: ql = r_mean + self.gamma * (1.0 - p_done) * v_lower[successor_nodes[0, a, 0]]
                else:
                    p_hat, v_next_l = p_buf[:k_max], v_l_buf[:k_max]
                    for k in range(k_max):
                        nid = successor_nodes[0, a, k]
                        p_hat[k], v_next_l[k] = successor_counts[0, a, k] / n_sa, v_lower[nid]
                    ql = r_mean + self.gamma * (1.0 - p_done) * min_expectation_under_constraint(v_next_l, p_hat, math.log(n_total_root + 1) / n_sa, use_newton=self.use_newton)
            if ql > max_ql: max_ql, best_a = ql, a
        return best_a, {"root_v": float(v_lower[0]), "nodes": int(final_max_n)}

    def get_name(self) -> str: return f"gbop_traj{self.trajectories}"
    def print_info(self): print(f"[GBOP] strategy={self.strategy}, trajectories={self.trajectories}, horizon={self.horizon}")
