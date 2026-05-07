import pytest
import numpy as np
from mcts_forest.utils.registry import REGISTRY

def test_numba_rollout_equivalence():
    """
    Statistically verifies that the procedural rollout_fn is mathematically equivalent to env.step()
    """
    num_trials = 10000
    env_name = "frozenlake8x8_slip"
    env = REGISTRY.get_env(env_name)
    
    # Get procedural dynamics
    _, rollout_fn, params = env.get_procedural_dynamics()
    
    init_state = env.reset()
    gamma = 0.99
    rollout_limit = 50
    
    # --- 1. Collection via Standard env.step() ---
    std_rewards = []
    for _ in range(num_trials):
        env.set_state(init_state)
        total_reward = 0.0
        curr_gamma = 1.0
        term, trunc = False, False
        steps = 0
        while not (term or trunc) and steps < rollout_limit:
            action = np.random.randint(0, env.action_space_size)
            _, r, term, trunc, _ = env.step(action)
            total_reward += curr_gamma * r
            curr_gamma *= gamma
            steps += 1
        std_rewards.append(total_reward)
    
    # --- 2. Collection via procedural rollout_fn ---
    nb_rewards = []
    # Warmup JIT
    rollout_fn(init_state, *params, rollout_limit, gamma)
    
    for _ in range(num_trials):
        r = rollout_fn(init_state, *params, rollout_limit, gamma)
        nb_rewards.append(r)
    
    # --- 3. Statistical Comparison ---
    std_mean = np.mean(std_rewards)
    std_std = np.std(std_rewards)
    nb_mean = np.mean(nb_rewards)
    nb_std = np.std(nb_rewards)
    
    combined_sem = np.sqrt((std_std**2 + nb_std**2) / num_trials)
    
    if combined_sem > 0:
        z_score = abs(std_mean - nb_mean) / combined_sem
    else:
        z_score = 0.0 if abs(std_mean - nb_mean) < 1e-7 else float('inf')
    
    assert z_score < 3.0, (
        f"Rollout mismatch. "
        f"Std Mean: {std_mean:.5f}, Numba Mean: {nb_mean:.5f}, Z-score: {z_score:.4f}"
    )

@pytest.mark.parametrize("env_name", ["taxi", "taxi_rain"])
def test_taxi_rollout_equivalence(env_name):
    """
    Verifies that procedural rollout_fn is mathematically equivalent to env.step() for Taxi variants.
    """
    num_trials = 2000
    env = REGISTRY.get_env(env_name)
    _, rollout_fn, params = env.get_procedural_dynamics()
    
    init_state = env.reset()
    gamma = 0.99
    rollout_limit = 50
    
    # 1. Std
    std_rewards = []
    for _ in range(num_trials):
        env.set_state(init_state)
        total_reward = 0.0
        curr_gamma = 1.0
        term, trunc = False, False
        steps = 0
        while not (term or trunc) and steps < rollout_limit:
            action = np.random.randint(0, env.action_space_size)
            _, r, term, trunc, _ = env.step(action)
            total_reward += curr_gamma * r
            curr_gamma *= gamma
            steps += 1
        std_rewards.append(total_reward)
    
    # 2. Procedural
    nb_rewards = []
    for _ in range(num_trials):
        nb_rewards.append(rollout_fn(init_state, *params, rollout_limit, gamma))
    
    # 3. Statistical Comparison
    std_mean, std_std = np.mean(std_rewards), np.std(std_rewards)
    nb_mean, nb_std = np.mean(nb_rewards), np.std(nb_rewards)
    
    combined_sem = np.sqrt((std_std**2 + nb_std**2) / num_trials)
    if combined_sem > 0:
        z_score = abs(std_mean - nb_mean) / combined_sem
    else:
        z_score = 0.0 if abs(std_mean - nb_mean) < 1e-7 else float('inf')
        
    assert z_score < 3.0, (
        f"Rollout mismatch for {env_name}. "
        f"Std Mean: {std_mean:.5f}, Numba Mean: {nb_mean:.5f}, Z-score: {z_score:.4f}"
    )
