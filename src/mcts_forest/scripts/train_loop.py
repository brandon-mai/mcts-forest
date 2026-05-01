import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from mcts_forest.core.base import TorchModelAdapter, ExperienceBuffer
from mcts_forest.core.gsp_alphazero import GSPAlphaZero, GSPAlphaZeroNet
from mcts_forest.core.gsp_muzero import GSPMuZero, GSPMuZeroNet
from mcts_forest.core.gsp_stochastic_muzero import GSPStochasticMuZero, GSPStochasticMuZeroNet
from mcts_forest.envs.gym_adapter import GymAdapter

def get_algorithm_components(algo_name):
    if algo_name == "gsp_alphazero": return GSPAlphaZero, GSPAlphaZeroNet
    if algo_name == "gsp_muzero": return GSPMuZero, GSPMuZeroNet
    if algo_name == "gsp_stochastic_muzero": return GSPStochasticMuZero, GSPStochasticMuZeroNet
    raise ValueError(f"Algorithm {algo_name} not supported.")

def self_play_iteration(algo_class, model_adapter, it_idx, num_games=50, sims=400):
    env = GymAdapter("FrozenLake-v1", is_slippery=True)
    solver = algo_class(env, model_adapter, simulation_limit=sims)
    buffer = ExperienceBuffer()
    print(f"--- [Self-Play] {num_games} games ---")
    for _ in tqdm(range(num_games)):
        state = env.reset(); episode_history, done = [], False
        while not done:
            action, info = solver.search(state)
            pi = info["action_visits"].astype(np.float32); pi /= pi.sum()
            episode_history.append((state, pi))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        buffer.add([(s, p, reward) for s, p in episode_history])
    return buffer

def train_iteration(model_adapter, buffer, epochs=5, batch_size=64, lr=0.001):
    print(f"--- [Training] {epochs} epochs ---")
    optimizer = optim.Adam(model_adapter.model.parameters(), lr=lr)
    model_adapter.model.train()
    one_hot_f = lambda s: np.eye(16)[s].astype(np.float32)
    steps = 0
    for _ in range(epochs):
        for states, target_pi, target_z in buffer.get_batches(batch_size, input_transform=one_hot_f):
            s_t = torch.tensor(states, dtype=torch.float32).to(model_adapter.device)
            pi_t = torch.tensor(target_pi, dtype=torch.float32).to(model_adapter.device)
            z_t = torch.tensor(target_z, dtype=torch.float32).to(model_adapter.device).unsqueeze(1)
            optimizer.zero_grad()
            pi_logits, v_pred = model_adapter.model(s_t)
            # Standard AlphaZero style loss
            value_loss = F.mse_loss(v_pred, z_t)
            policy_loss = -torch.mean(torch.sum(pi_t * F.log_softmax(pi_logits, dim=-1), dim=-1))
            (value_loss + policy_loss).backward()
            optimizer.step()
            steps += 1
    model_adapter.model.eval()
    return steps

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="gsp_alphazero")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games_per_it", type=int, default=1)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    algo_class, net_class = get_algorithm_components(args.algo)
    checkpoint_dir = f"checkpoints/{args.algo}"; os.makedirs(checkpoint_dir, exist_ok=True)
    adapter = TorchModelAdapter(net_class()); latest_pt = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_pt): adapter.load(latest_pt)

    it, total_steps = 1, 0
    for it_cnt in range(args.iterations):
        curr_it = it + it_cnt
        buffer = self_play_iteration(algo_class, adapter, curr_it, num_games=args.games_per_it, sims=args.sims)
        steps = train_iteration(adapter, buffer, epochs=args.epochs)
        total_steps += steps
        v_name = f"v{curr_it}_{total_steps}"
        adapter.save(os.path.join(checkpoint_dir, f"{v_name}.pt")); adapter.save(latest_pt)
        print(f"*** Iteration {curr_it} Complete | Total Steps: {total_steps} ***\n")

if __name__ == "__main__": main()
