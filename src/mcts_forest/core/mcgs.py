import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
from mcts_forest.core.base import sample_discrete_transition, random_rollout_discrete
from mcts_forest.utils.visualizer import SurrogateNode

@njit
def mcgs_core(initial_state, transitions, rewards, dones, probs_cum, c, horizon, gamma, rollout_limit, simulation_limit, max_n, visit_count, q_hat, action_visits, state_to_node, node_to_state, curr_max_n, child_nodes):
    for _ in range(simulation_limit):
        curr_s = initial_state
        curr_node = np.int32(0)
        
        p_nodes = np.zeros(horizon + 1, dtype=np.int32)
        p_a = np.zeros(horizon + 1, dtype=np.int32)
        p_r = np.zeros(horizon + 1, dtype=np.float32)
        p_len = 0
        
        v_leaf = -100.0
        
        while p_len < horizon:
            best_a = -1
            max_u = -1e9
            n_p = visit_count[curr_node]
            ln_p = math.log(float(max(1, n_p)))
            # Selection
            found_unvisited = False
            for a in range(q_hat.shape[1]):
                if action_visits[curr_node, a] == 0:
                    best_a = a
                    found_unvisited = True
                    break
            
            if not found_unvisited:
                for a in range(q_hat.shape[1]):
                    n_sa = action_visits[curr_node, a]
                    val = q_hat[curr_node, a] + c * math.sqrt(ln_p / n_sa)
                    # val = q_hat[curr_node, a] + c * (n_p ** 0.25 / n_sa ** 0.5)
                    if val > max_u:
                        max_u = val
                        best_a = a
            
            # Step
            s_nxt, r, d = sample_discrete_transition(curr_s, best_a, transitions, rewards, dones, probs_cum)
            
            p_nodes[p_len] = curr_node
            p_a[p_len] = best_a
            p_r[p_len] = r
            p_len += 1
            
            s_nxt_i = np.int32(s_nxt)
            if s_nxt_i not in state_to_node:
                if curr_max_n < max_n:
                    new_node = np.int32(curr_max_n)
                    state_to_node[s_nxt_i] = new_node
                    node_to_state[new_node] = s_nxt_i
                    curr_max_n += 1
                    child_nodes[curr_node, best_a] = new_node
                    v_leaf = random_rollout_discrete(s_nxt, transitions, rewards, dones, probs_cum, rollout_limit, gamma)
                    break
                else:
                    v_leaf = random_rollout_discrete(s_nxt, transitions, rewards, dones, probs_cum, rollout_limit, gamma)
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
            v_leaf = random_rollout_discrete(curr_s, transitions, rewards, dones, probs_cum, rollout_limit, gamma)

        # Backpropagation
        v_nxt = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r = p_nodes[i], p_a[i], p_r[i]
            action_visits[n_id, a] += 1
            visit_count[n_id] += 1
            q_hat[n_id, a] += (r + gamma * v_nxt - q_hat[n_id, a]) / action_visits[n_id, a]
            # Bootstrap
            v_nxt = float(-1e9)
            for aa in range(q_hat.shape[1]):
                if q_hat[n_id, aa] > v_nxt:
                    v_nxt = q_hat[n_id, aa]
    
    return curr_max_n

class MCGS:
    def __init__(self, env, c=1.414, horizon=100, gamma=0.99, rollout_limit=100, simulation_limit=1000, **kwargs):
        self.env = env
        self.c = c
        self.horizon = horizon
        self.gamma = gamma
        self.rollout_limit = rollout_limit
        self.simulation_limit = simulation_limit
        self.dynamics = env.get_numba_dynamics()
        self.max_n = 20000

    def search(self, initial_state):
        max_n = self.max_n
        v_count = np.zeros(max_n, dtype=np.int32)
        q_hat = np.full((max_n, self.dynamics[0].shape[1]), -100.0, dtype=np.float32)
        a_visits = np.zeros((max_n, self.dynamics[0].shape[1]), dtype=np.int32)
        state_to_node = NumbaDict.empty(numba.int32, numba.int32)
        node_to_state = np.zeros(max_n, dtype=np.int32)
        child_nodes = np.full((max_n, self.dynamics[0].shape[1]), -1, dtype=np.int32)
        
        state_to_node[np.int32(initial_state)] = np.int32(0)
        node_to_state[0] = np.int32(initial_state)
        
        final_max_n = mcgs_core(
            initial_state, *self.dynamics, self.c, self.horizon, self.gamma, self.rollout_limit, self.simulation_limit, 
            max_n, v_count, q_hat, a_visits, state_to_node, node_to_state, 1, child_nodes
        )
        
        best_a = int(np.argmax(q_hat[0]))
        root = None
        # root = SurrogateNode(0, v_count, q_hat, child_nodes, a_visits=a_visits)
        return best_a, {"root": root}

    def get_name(self):
        return f"mcgs_sim{self.simulation_limit}"
