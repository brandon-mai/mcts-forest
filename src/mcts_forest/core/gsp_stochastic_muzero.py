import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, Any
from mcts_forest.core.base import TorchModelAdapter, ExperienceBuffer
from mcts_forest.core.gsp_muzero import GSPMuZeroNet, MuZeroNode

class GSPStochasticMuZeroNet(GSPMuZeroNet):
    def __init__(self, obs_dim=16, n_actions=4, latent_dim=64, num_outcomes=3):
        super().__init__(obs_dim, n_actions, latent_dim)
        self.num_outcomes = num_outcomes
        self.afterstate_h = nn.Sequential(nn.Linear(latent_dim + n_actions, latent_dim), nn.ReLU())
        self.outcome_dynamics = nn.Sequential(nn.Linear(latent_dim + num_outcomes, latent_dim), nn.ReLU())
        self.chance_head = nn.Linear(latent_dim, num_outcomes)

    def dynamics(self, h, a_oh): return self.afterstate_h(torch.cat([h, a_oh], dim=-1))
    def sample_outcome(self, h_after):
        probs = F.softmax(self.chance_head(h_after), dim=-1)
        o_idx = torch.multinomial(probs, 1).item()
        o_oh = torch.zeros(1, self.num_outcomes).to(h_after.device); o_oh[0, o_idx] = 1.0
        h_next = self.outcome_dynamics(torch.cat([h_after, o_oh], dim=-1))
        return h_next, torch.sigmoid(self.reward_head(h_next)), o_idx

class GSPStochasticMuZeroNode(MuZeroNode):
    def __init__(self, latent, prior):
        super().__init__(latent, prior)
        self.chance_nodes = {} 

class ChanceNode:
    def __init__(self):
        self.outcomes = {}; self.visit_count = 0

class GSPStochasticMuZero:
    def __init__(self, env, model_adapter: TorchModelAdapter, c_puct=1.25, p=2.0, gamma=0.99, simulation_limit=100, **kwargs):
        self.model = model_adapter.model.to(model_adapter.device)
        self.device, self.c_puct, self.p, self.gamma, self.simulation_limit = model_adapter.device, c_puct, p, gamma, simulation_limit
        self.n_actions = 4

    def search(self, state_idx: int) -> Tuple[int, Dict[str, Any]]:
        self.model.eval()
        n_sim = self.simulation_limit
        horizon = int(math.ceil(math.log(n_sim) / (2 * math.log(1.0 / self.gamma)))) if self.gamma < 1.0 else 100
        
        with torch.no_grad():
            s_oh = torch.zeros(1, 16).to(self.device); s_oh[0, state_idx] = 1.0
            h0 = self.model.represent(s_oh)
            pi_log, _ = self.model.predict(h0)
            root = GSPStochasticMuZeroNode(h0, 0.0)
            root.pi = F.softmax(pi_log, dim=-1).cpu().numpy()[0]
            
            for _ in range(n_sim):
                node, search_path, h_depth = root, [root], 0
                while node.chance_nodes and h_depth < horizon:
                    best_s, best_a = -1e18, -1
                    sqrt_n = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0
                    for a, chance in node.chance_nodes.items():
                        q = sum(o.value_sum for o in chance.outcomes.values()) / chance.visit_count if chance.visit_count > 0 else 0.0
                        score = q + self.c_puct * node.pi[a] * sqrt_n / (1 + chance.visit_count)
                        if score > best_s: best_s, best_a = score, a
                    
                    h_after = self.model.dynamics(node.latent, self._oh_act(best_a))
                    h_next, rew, o_idx = self.model.sample_outcome(h_after)
                    chance = node.chance_nodes[best_a]
                    if o_idx not in chance.outcomes:
                        p_log, _ = self.model.predict(h_next)
                        new_node = GSPStochasticMuZeroNode(h_next, 0.0)
                        new_node.pi, new_node.reward = F.softmax(p_log, dim=-1).cpu().numpy()[0], rew.item()
                        chance.outcomes[o_idx] = new_node
                    
                    chance.visit_count += 1
                    node = chance.outcomes[o_idx]
                    search_path.append(node)
                    h_depth += 1
                    if not node.chance_nodes and node.visit_count > 0: break
                
                if h_depth < horizon and not node.chance_nodes:
                    for a in range(self.n_actions): node.chance_nodes[a] = ChanceNode()
                    _, v = self.model.predict(node.latent); v_leaf = v.item()
                else: _, v = self.model.predict(node.latent); v_leaf = v.item()

                v_back = v_leaf
                for i in range(len(search_path)-1, -1, -1):
                    curr = search_path[i]
                    curr.visit_count += 1
                    if not curr.chance_nodes: curr.value_sum = float(v_back) * curr.visit_count
                    else:
                        # Calculate q values for chance nodes and find q_min
                        qcs = []
                        for chance in curr.chance_nodes.values():
                            if chance.visit_count > 0:
                                q_c = sum((o.visit_count/chance.visit_count)*(o.reward + self.gamma*o.value()) for o in chance.outcomes.values())
                                qcs.append(q_c)
                        
                        curr_shift = max(0.0, -min(qcs)) + 1.0 if qcs else 1.0
                        
                        w_sum = 0.0
                        for a, chance in curr.chance_nodes.items():
                            if chance.visit_count > 0:
                                q_c = sum((o.visit_count/chance.visit_count)*(o.reward + self.gamma*o.value()) for o in chance.outcomes.values())
                                q_val = q_c + curr_shift
                                if q_val < 0.0: q_val = 0.0
                                w_sum += (chance.visit_count/curr.visit_count) * (q_val ** self.p)
                        v_back = (w_sum**(1.0/self.p))-1.0
                        curr.value_sum = float(v_back)*curr.visit_count
                        
        counts = np.array([root.chance_nodes[a].visit_count for a in range(self.n_actions)])
        return int(np.argmax(counts)), {"action_visits": counts}

    def _oh_act(self, a):
        oh = torch.zeros(1, self.n_actions).to(self.device); oh[0, a] = 1.0
        return oh
