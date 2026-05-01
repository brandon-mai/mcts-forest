import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import numba
from tqdm import tqdm
from mcts_forest.core.base import TorchModelAdapter, ExperienceBuffer
from mcts_forest.utils.registry import REGISTRY

# 1. Kaggle CPU Optimization
numba.config.NUMBA_NUM_THREADS = 4

def get_algorithm_components(algo_name):
    """Maps solver names to their core class and network class."""
    from mcts_forest.core.gsp_alphazero import GSPAlphaZero, GSPAlphaZeroNet
    from mcts_forest.core.gsp_muzero import GSPMuZero, GSPMuZeroNet
    from mcts_forest.core.gsp_stochastic_muzero import GSPStochasticMuZero, GSPStochasticMuZeroNet
    from mcts_forest.core.alphazero import AlphaZero, AlphaZeroNet
    from mcts_forest.core.muzero import MuZero, MuZeroNet
    
    mapping = {
        "gsp_alphazero": (GSPAlphaZero, GSPAlphaZeroNet),
        "alphazero": (AlphaZero, AlphaZeroNet),
        "gsp_muzero": (GSPMuZero, GSPMuZeroNet),
        "muzero": (MuZero, MuZeroNet),
        "gsp_stochastic_muzero": (GSPStochasticMuZero, GSPStochasticMuZeroNet),
    }
    if algo_name not in mapping:
        raise ValueError(f"Algorithm {algo_name} not supported.")
    return mapping[algo_name]

from mcts_forest.utils.stats import bootstrap_stats

def evaluate_iteration(algo_class, model_adapter, env_name, num_games=10, sims=100, prefix="eval"):
    env = REGISTRY.get_env(env_name)
    solver = algo_class(env, model_adapter, simulation_limit=sims)
    rewards, successes = [], []
    print(f"--- [{prefix.upper()}] {num_games} games ---")
    for _ in range(num_games):
        state = env.reset(); done = False; total_r = 0; is_success = False
        while not done:
            action, _ = solver.search(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            if terminated: is_success = True
            done = terminated or truncated
        rewards.append(total_r)
        successes.append(float(is_success))
    
    r_mean, r_std = bootstrap_stats(np.array(rewards))
    s_mean, s_std = bootstrap_stats(np.array(successes))
    
    print(f"--> {prefix} Reward: {r_mean:.4f} ± {r_std:.4f} | Success: {s_mean*100:.2f}% ± {s_std*100:.2f}%")
    
    if "wandb" in globals() and wandb.run:
        wandb.log({
            f"{prefix}/reward_mean": r_mean,
            f"{prefix}/reward_std": r_std,
            f"{prefix}/success_rate": s_mean,
            f"{prefix}/success_std": s_std
        })
    return r_mean

def self_play_iteration(algo_class, model_adapter, env_name, num_games=50, sims=400, prefix="Self-Play"):
    env = REGISTRY.get_env(env_name)
    solver = algo_class(env, model_adapter, simulation_limit=sims)
    buffer = ExperienceBuffer()
    print(f"--- [{prefix}] {num_games} games ---")
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

def train_iteration(model_adapter, buffer, obs_dim, epochs=5, batch_size=64, lr=0.001, use_amp=False, prefix="train"):
    print(f"--- [Training {prefix}] {epochs} epochs ---")
    optimizer = optim.Adam(model_adapter.model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    model_adapter.model.train()
    steps = 0
    for _ in range(epochs):
        for states, target_pi, target_z in buffer.get_batches(batch_size):
            if states.ndim == 1:
                states = np.eye(obs_dim)[states].astype(np.float32)
            s_t = torch.tensor(states, dtype=torch.float32).to(model_adapter.device)
            pi_t = torch.tensor(target_pi, dtype=torch.float32).to(model_adapter.device)
            z_t = torch.tensor(target_z, dtype=torch.float32).to(model_adapter.device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=model_adapter.device.type, enabled=use_amp):
                pi_logits, v_pred = model_adapter.model(s_t)
                value_loss = F.mse_loss(v_pred, z_t)
                policy_loss = -torch.mean(torch.sum(pi_t * F.log_softmax(pi_logits, dim=-1), dim=-1))
                loss = value_loss + policy_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            if "wandb" in globals() and wandb.run:
                wandb.log({f"{prefix}/loss": loss.item()})
    model_adapter.model.eval()
    return steps

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="gsp_alphazero")
    parser.add_argument("--env", type=str, default="frozenlake_slip")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games_per_it", type=int, default=1)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.wandb:
        global wandb
        import wandb
        wandb.init(project="mcts-forest", config=vars(args))

    temp_env = REGISTRY.get_env(args.env)
    obs_dim = temp_env.observation_space.n if hasattr(temp_env.observation_space, 'n') else temp_env.observation_space.shape[0]
    n_actions = temp_env.action_space.n
    num_outcomes = 1
    dynamics = temp_env.get_numba_dynamics()
    if dynamics is not None: num_outcomes = dynamics[0].shape[2]

    # Solver Mapping
    gsp_algo_name = args.algo if args.algo.startswith("gsp_") else f"gsp_{args.algo}"
    baseline_algo_name = gsp_algo_name.replace("gsp_", "")
    
    gsp_class, gsp_net = get_algorithm_components(gsp_algo_name)
    base_class, base_net = get_algorithm_components(baseline_algo_name)
    
    net_args = {"obs_dim": obs_dim, "n_actions": n_actions}
    if "stochastic" in gsp_algo_name: net_args["num_outcomes"] = num_outcomes
    
    # Adapters for 3 variants
    gsp_adapter = TorchModelAdapter(gsp_net(**net_args))
    base_adapter = TorchModelAdapter(base_net(**net_args))
    cross_adapter = TorchModelAdapter(gsp_net(**net_args)) # Same net as GSP, but trained on Baseline data

    dirs = {
        "gsp": f"checkpoints/{gsp_algo_name}/{args.env}",
        "baseline": f"checkpoints/{baseline_algo_name}/{args.env}",
        "cross": f"checkpoints/cross_{gsp_algo_name}/{args.env}"
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    def load_latest(adapter, d):
        lp = os.path.join(d, "latest.pt")
        if os.path.exists(lp): adapter.load(lp)

    load_latest(gsp_adapter, dirs["gsp"])
    load_latest(base_adapter, dirs["baseline"])
    load_latest(cross_adapter, dirs["cross"])

    for it_cnt in range(args.iterations):
        curr_it = it_cnt + 1
        
        # 1. Self-Play
        buffer_gsp = self_play_iteration(gsp_class, gsp_adapter, args.env, num_games=args.games_per_it, sims=args.sims, prefix="Self-Play GSP")
        buffer_base = self_play_iteration(base_class, base_adapter, args.env, num_games=args.games_per_it, sims=args.sims, prefix="Self-Play Baseline")
        
        # 2. Training
        train_iteration(gsp_adapter, buffer_gsp, obs_dim, epochs=args.epochs, use_amp=args.use_amp, prefix="gsp")
        train_iteration(base_adapter, buffer_base, obs_dim, epochs=args.epochs, use_amp=args.use_amp, prefix="baseline")
        train_iteration(cross_adapter, buffer_base, obs_dim, epochs=args.epochs, use_amp=args.use_amp, prefix="cross")
        
        # Save
        gsp_adapter.save(os.path.join(dirs["gsp"], "latest.pt"))
        base_adapter.save(os.path.join(dirs["baseline"], "latest.pt"))
        cross_adapter.save(os.path.join(dirs["cross"], "latest.pt"))
            
        # 3. Evaluation
        evaluate_iteration(gsp_class, gsp_adapter, args.env, num_games=10, sims=args.sims, prefix="gsp")
        evaluate_iteration(base_class, base_adapter, args.env, num_games=10, sims=args.sims, prefix="baseline")
        evaluate_iteration(gsp_class, cross_adapter, args.env, num_games=10, sims=args.sims, prefix="cross")
            
        print(f"*** Iteration {curr_it} Complete ***\n")

if __name__ == "__main__": main()
