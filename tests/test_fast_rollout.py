import pytest
import numpy as np
from mcts_forest.utils.registry import REGISTRY
from mcts_forest.core.base import random_rollout_discrete as numba_rollout

def skew_frozenlake_probabilities(env, success_rate: float):
    """
    Modifies the FrozenLake environment's P table to have a specific success rate.
    The remaining probability (1.0 - success_rate) is split between the other outcomes.
    """
    if not hasattr(env.env, 'P'):
        return
        
    for s in env.env.P:
        for a in env.env.P[s]:
            outcomes = env.env.P[s][a]
            if len(outcomes) <= 1:
                continue
            
            # Identify the outcomes. In FrozenLake slippery, there are usually 3.
            # We set the first one to success_rate and split the rest.
            num_others = len(outcomes) - 1
            other_prob = (1.0 - success_rate) / num_others
            
            new_outcomes = []
            for i, (p, ns, r, d) in enumerate(outcomes):
                p_new = success_rate if i == 0 else other_prob
                new_outcomes.append((p_new, ns, r, d))
            
            env.env.P[s][a] = new_outcomes

@pytest.mark.parametrize("success_rate", [0.1, 0.5, 0.9])
def test_numba_rollout_equivalence(success_rate):
    """
    Statistically verifies that numba_rollout is mathematically equivalent to env.step() 
    even with custom, non-uniform transition probabilities.
    """
    num_trials = 10000
    env_name = "frozenlake8x8_slip"
    env = REGISTRY.get_env(env_name)
    
    # Skew the probabilities in the underlying Gym env
    skew_frozenlake_probabilities(env, success_rate)
    
    # Refresh dynamics matrices to reflect the skewed P table
    dynamics = env.get_numba_dynamics()
    assert dynamics is not None, "Dynamics extraction failed"
    
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
            # We use the same random policy (uniform random actions) for both
            action = np.random.randint(0, env.action_space_size)
            _, r, term, trunc, _ = env.step(action)
            total_reward += curr_gamma * r
            curr_gamma *= gamma
            steps += 1
        std_rewards.append(total_reward)
    
    # --- 2. Collection via numba_rollout() ---
    nb_rewards = []
    # Warmup JIT
    numba_rollout(init_state, dynamics[0], dynamics[1], dynamics[2], dynamics[3], rollout_limit, gamma, 0.0, 1.0)
    
    for _ in range(num_trials):
        r = numba_rollout(init_state, *dynamics, rollout_limit, gamma, 0.0, 1.0)
        nb_rewards.append(r)
    
    # --- 3. Statistical Comparison ---
    std_mean = np.mean(std_rewards)
    std_std = np.std(std_rewards)
    nb_mean = np.mean(nb_rewards)
    nb_std = np.std(nb_rewards)
    
    # Standard Error of the Mean Difference
    combined_sem = np.sqrt((std_std**2 + nb_std**2) / num_trials)
    
    if combined_sem > 0:
        z_score = abs(std_mean - nb_mean) / combined_sem
    else:
        # Zero variance: means must match exactly
        z_score = 0.0 if abs(std_mean - nb_mean) < 1e-7 else float('inf')
    
    # We use a threshold of 3.0 (99.7% confidence). 
    # Statistically, there's a small chance of false failure, but 3.0 is safe for CI.
    assert z_score < 3.0, (
        f"Rollout mismatch for success_rate={success_rate}. "
        f"Std Mean: {std_mean:.5f}, Numba Mean: {nb_mean:.5f}, Z-score: {z_score:.4f}"
    )

@pytest.mark.parametrize("env_name", ["taxi", "taxi_rain"])
def test_taxi_rollout_equivalence(env_name):
    """
    Verifies that numba_rollout is mathematically equivalent to env.step() for Taxi variants.
    """
    num_trials = 2000
    env = REGISTRY.get_env(env_name)
    dynamics = env.get_numba_dynamics()
    assert dynamics is not None, "Dynamics extraction failed"
    
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
    
    # 2. Numba
    nb_rewards = []
    for _ in range(num_trials):
        nb_rewards.append(numba_rollout(init_state, *dynamics, rollout_limit, gamma, 0.0, 1.0))
    
    # 3. Statistical Comparison
    std_mean, std_std = np.mean(std_rewards), np.std(std_rewards)
    nb_mean, nb_std = np.mean(nb_rewards), np.std(nb_rewards)
    
    combined_sem = np.sqrt((std_std**2 + nb_std**2) / num_trials)
    if combined_sem > 0:
        z_score = abs(std_mean - nb_mean) / combined_sem
    else:
        z_score = 0.0 if abs(std_mean - nb_mean) < 1e-7 else float('inf')
        
    print(f"\n{env_name} - Std Mean: {std_mean:.2f}, Numba Mean: {nb_mean:.2f}, Std StdDev: {std_std:.2f}")
    assert z_score < 3.0, (
        f"Rollout mismatch for {env_name}. "
        f"Std Mean: {std_mean:.5f}, Numba Mean: {nb_mean:.5f}, Z-score: {z_score:.4f}"
    )
