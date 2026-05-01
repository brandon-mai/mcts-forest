import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, Any
from mcts_forest.core.base import TorchModelAdapter, ExperienceBuffer

class MuZeroNet(nn.Module):
    def __init__(self, obs_dim=16, n_actions=4, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.representation = nn.Sequential(nn.Linear(obs_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.dynamics_h = nn.Sequential(nn.Linear(latent_dim + n_actions, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.reward_head = nn.Linear(latent_dim, 1)
        self.policy_head = nn.Linear(latent_dim, n_actions)
        self.value_head = nn.Linear(latent_dim, 1)

    def represent(self, s_oh): return self.representation(s_oh)
    def dynamics(self, h, a_oh):
        h_next = self.dynamics_h(torch.cat([h, a_oh], dim=-1))
        return h_next, torch.sigmoid(self.reward_head(h_next))
    def predict(self, h): return self.policy_head(h), torch.sigmoid(self.value_head(h))
    def forward(self, s_oh): return self.predict(self.represent(s_oh))

class MuZeroNode:
    def __init__(self, latent, prior):
        self.latent, self.prior = latent, prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
    def value(self): return float(self.value_sum / self.visit_count) if self.visit_count > 0 else 0.0

class MuZero:
    def __init__(self, env, model_adapter: TorchModelAdapter, c_puct=1.25, gamma=0.99, simulation_limit=100, **kwargs):
        self.model = model_adapter.model.to(model_adapter.device)
        self.device, self.c_puct, self.gamma, self.simulation_limit = model_adapter.device, c_puct, gamma, simulation_limit
        self.obs_dim = env.observation_space.n if hasattr(env.observation_space, 'n') else env.observation_space.shape[0]
        self.n_actions = env.action_space.n

    def search(self, state_idx: int) -> Tuple[int, Dict[str, Any]]:
        self.model.eval()
        n_sim = self.simulation_limit
        horizon = 100
        
        with torch.no_grad():
            # Handle discrete state space (FrozenLake style)
            s_oh = torch.zeros(1, self.obs_dim).to(self.device); s_oh[0, state_idx] = 1.0
            h0 = self.model.represent(s_oh)
            pi_logits, _ = self.model.predict(h0)
            root = MuZeroNode(h0, 0.0)
            root.pi = F.softmax(pi_logits, dim=-1).cpu().numpy()[0]
            
            for _ in range(n_sim):
                node, search_path, h_depth = root, [root], 0
                while node.children and h_depth < horizon:
                    best_s, best_a = -1e18, -1
                    sqrt_n = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0
                    for a, child in node.children.items():
                        score = child.value() + self.c_puct * node.pi[a] * sqrt_n / (1 + child.visit_count)
                        if score > best_s: best_s, best_a = score, a
                    node = node.children[best_a]
                    search_path.append(node)
                    h_depth += 1
                
                if h_depth < horizon:
                    if len(search_path) > 1:
                        parent = search_path[-2]
                        a = [act for act, n in parent.children.items() if n == node][0]
                        a_oh = torch.zeros(1, self.n_actions).to(self.device); a_oh[0, a] = 1.0
                        node.latent, rew = self.model.dynamics(parent.latent, a_oh)
                        node.reward = rew.item()
                    pi_log, v = self.model.predict(node.latent)
                    node.pi, v_leaf = F.softmax(pi_log, dim=-1).cpu().numpy()[0], v.item()
                    for a in range(self.n_actions): node.children[a] = MuZeroNode(None, node.pi[a])
                else: _, v = self.model.predict(node.latent); v_leaf = v.item()

                v_back = v_leaf
                for i in range(len(search_path)-1, -1, -1):
                    curr = search_path[i]
                    curr.visit_count += 1
                    # Standard Arithmetic Mean Backup
                    if not curr.children: 
                        curr.value_sum = float(v_back) * curr.visit_count
                    else:
                        # Mean of children values
                        total_q = 0.0
                        count = 0
                        for child in curr.children.values():
                            if child.visit_count > 0:
                                q = child.reward + self.gamma * child.value()
                                total_q += q
                                count += 1
                        v_back = total_q / count if count > 0 else v_leaf
                        curr.value_sum = float(v_back) * curr.visit_count
                    
        counts = np.array([root.children[a].visit_count for a in range(self.n_actions)])
        return int(np.argmax(counts)), {"action_visits": counts}
