import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict as NumbaDict
from typing import Tuple, Dict, Any
from mcts_forest.core.base import sample_discrete_transition, random_rollout_discrete


@njit(cache=True)
def ments_core(n_sim, horizon, tau, c, gamma, rollout_limit, dynamics, 
               visit_count, q_hat, v_node, action_visits, child_nodes, node_states, node_counter, 
               reward_offset, reward_scale):
    transitions, rewards, dones, probs_cum = dynamics
    n_actions = q_hat.shape[1]
    
    p_nodes = np.empty(horizon + 1, dtype=np.int32)
    p_a = np.empty(horizon + 1, dtype=np.int32)
    p_r_int = np.empty(horizon + 1, dtype=np.float32)
    
    sim_count = 0
    while sim_count < n_sim:
        curr_node, curr_h = 0, 0
        p_len = 0
        v_leaf = 0.0
        
        # 1. Traversal
        while curr_h < horizon:
            s_idx = node_states[curr_node]
            
            # MENTS Stochastic Selection (Boltzmann)
            # Literature MENTS always uses softmax once a node is fully expanded
            qs = q_hat[curr_node]
            max_q = np.max(qs)
            sum_exp = 0.0
            for i in range(n_actions):
                sum_exp += math.exp((qs[i] - max_q) / tau)
            
            # Lambda coefficient (Uniform mix for exploration)
            n_total = float(visit_count[curr_node])
            lambda_coeff = min(1.0, (c * n_actions) / math.log(n_total + 1.0 + 1e-10))
            
            # Sample action
            r_val = np.random.random()
            cum_p = 0.0
            a = n_actions - 1
            for i in range(n_actions):
                p_softmax = math.exp((qs[i] - max_q) / tau) / sum_exp
                p_final = (1.0 - lambda_coeff) * p_softmax + (lambda_coeff / n_actions)
                cum_p += p_final
                if r_val < cum_p:
                    a = i
                    break
            
            # Step in environment
            next_s, r, done = sample_discrete_transition(s_idx, a, transitions, rewards, dones, probs_cum)
            r_int = (r + reward_offset) * reward_scale
            
            key = (np.int32(curr_node), np.int32(a), np.int32(next_s))
            p_nodes[p_len], p_a[p_len], p_r_int[p_len] = curr_node, a, r_int
            p_len += 1
            
            if done:
                v_leaf = 0.0
                break
            
            if key in child_nodes:
                curr_node = child_nodes[key]
                curr_h += 1
            else:
                # 2. FULL EXPANSION (Literature Faithful)
                # When we hit an unknown outcome, we create a new node and 
                # IMMEDIATELY initialize all its actions with rollouts
                new_node_id = node_counter[0]
                node_counter[0] += 1
                child_nodes[key] = new_node_id
                node_states[new_node_id] = next_s
                
                # Expand ALL actions for this new node immediately
                for act in range(n_actions):
                    # One sample per action for initialization
                    s_nxt_init, r_init, d_init = sample_discrete_transition(next_s, act, transitions, rewards, dones, probs_cum)
                    r_int_init = (r_init + reward_offset) * reward_scale
                    
                    if d_init:
                        v_child = 0.0
                    else:
                        v_child = random_rollout_discrete(s_nxt_init, transitions, rewards, dones, probs_cum, rollout_limit, gamma, reward_offset, reward_scale)
                    
                    # Update Q(new_node, act)
                    action_visits[new_node_id, act] = 1
                    q_hat[new_node_id, act] = r_int_init + gamma * v_child
                
                # Update V(new_node)
                qs_new = q_hat[new_node_id]
                mq = np.max(qs_new)
                se = 0.0
                for k in range(n_actions):
                    se += math.exp((qs_new[k] - mq) / tau)
                v_node[new_node_id] = mq + tau * math.log(se)
                visit_count[new_node_id] = n_actions
                
                v_leaf = v_node[new_node_id]
                sim_count += n_actions # Count the simulations used for expansion
                break
        else:
            v_leaf = v_node[curr_node]

        # 3. Backpropagation
        v_back = v_leaf
        for i in range(p_len - 1, -1, -1):
            n_id, a, r_int = p_nodes[i], p_a[i], p_r_int[i]
            action_visits[n_id, a] += 1
            q_hat[n_id, a] += (r_int + gamma * v_back - q_hat[n_id, a]) / action_visits[n_id, a]
            
            # Update V(s) = tau * log(sum(exp(Q/tau)))
            qs = q_hat[n_id]
            max_q = np.max(qs)
            sum_exp = 0.0
            for k in range(n_actions):
                sum_exp += math.exp((qs[k] - max_q) / tau)
            
            v_node[n_id] = max_q + tau * math.log(sum_exp)
            visit_count[n_id] += 1
            v_back = v_node[n_id]
            
        sim_count += 1


class MENTS:
    def __init__(self, env, tau=0.1, c=1.0, horizon=100, gamma=0.99, 
                 rollout_limit=100, simulation_limit=1000, 
                 internal_reward_offset=0.0, internal_reward_scale=1.0, init_q=0.0, **kwargs):
        self.env = env
        self.tau = tau
        self.c = c * internal_reward_scale
        self.horizon = horizon
        self.gamma = gamma
        self.rollout_limit = rollout_limit
        self.simulation_limit = simulation_limit
        self.reward_offset = internal_reward_offset
        self.reward_scale = internal_reward_scale
        self.init_q = init_q
        self.dynamics = env.get_numba_dynamics()

    def search(self, initial_state: int) -> Tuple[int, Dict[str, Any]]:
        # MENTS needs a generous node budget due to full expansion
        max_n = self.simulation_limit + 1000
        n_actions = self.dynamics[0].shape[1]
        
        visit_count = np.zeros(max_n, dtype=np.int32)
        q_hat = np.full((max_n, n_actions), self.init_q, dtype=np.float32)
        v_node = np.full(max_n, self.init_q, dtype=np.float32)
        action_visits = np.zeros((max_n, n_actions), dtype=np.int32)
        node_states = np.zeros(max_n, dtype=np.int32)
        node_counter = np.array([1], dtype=np.int32)
        
        node_states[0] = initial_state
        visit_count[0] = 1
        
        child_nodes = NumbaDict.empty(
            key_type=numba.types.Tuple((numba.int32, numba.int32, numba.int32)),
            value_type=numba.int32
        )
        
        # Initialize root with full expansion to match reference
        for act in range(n_actions):
            s_nxt, r, d = sample_discrete_transition(initial_state, act, *self.dynamics)
            r_int = (r + self.reward_offset) * self.reward_scale
            v_child = 0.0 if d else random_rollout_discrete(s_nxt, *self.dynamics, self.rollout_limit, self.gamma, self.reward_offset, self.reward_scale)
            q_hat[0, act] = r_int + self.gamma * v_child
            action_visits[0, act] = 1
        
        # Initial root V
        qs0 = q_hat[0]
        mq0 = np.max(qs0)
        se0 = 0.0
        for k in range(n_actions):
            se0 += math.exp((qs0[k] - mq0) / self.tau)
        v_node[0] = mq0 + self.tau * math.log(se0)
        visit_count[0] = n_actions

        ments_core(
            self.simulation_limit, self.horizon, self.tau, self.c, self.gamma, self.rollout_limit,
            self.dynamics, visit_count, q_hat, v_node, action_visits, child_nodes, node_states, node_counter,
            self.reward_offset, self.reward_scale
        )
        
        return int(np.argmax(q_hat[0])), {"root_v": float(v_node[0])}

    def get_name(self) -> str:
        return f"ments_sim{self.simulation_limit}"
