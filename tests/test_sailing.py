import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import mcts_forest.envs # Ensure registration
import numpy as np

def test_sailing_validity():
    """Gymnasium standard environment validity check."""
    env = gym.make("Sailing-v0")
    # check_env expects the raw env, and handles spaces/step logic
    check_env(env.unwrapped)

def test_sailing_determinism():
    """Verifies that the same seed and action sequence produce the same reward."""
    def run_ep(seed):
        env = gym.make("Sailing-v0", grid_size=5, time_limit=10)
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        actions = [0, 1, 2, 0, 1, 2] # R, DR, D ...
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            total_reward += r
            if term or trunc: break
        return total_reward

    r1 = run_ep(42)
    r2 = run_ep(42)
    r3 = run_ep(123)
    
    assert r1 == r2, "Same seed should result in same total reward"
    assert r1 != r3, "Different seed should likely result in different reward"

def test_sailing_wind_keep_probability():
    """Statistically verifies that wind keep probability is respected."""
    p_keep = 0.9
    env = gym.make("Sailing-v0", wind_keep_probability=p_keep, time_limit=2000)
    env.reset(seed=42)
    
    sequence = env.unwrapped.wind_sequence
    matches = 0
    for i in range(len(sequence) - 1):
        if sequence[i] == sequence[i+1]:
            matches += 1
    
    actual_prob = matches / (len(sequence) - 1)
    # With 2000 samples, 0.9 should be very close. 
    # Binomial distribution std dev for N=2000, p=0.9 is sqrt(2000*0.9*0.1) ~ 13.4. 
    # 3 sigma is ~40. 40/2000 = 0.02.
    assert abs(actual_prob - p_keep) < 0.05, f"Expected prob ~{p_keep}, got {actual_prob}"

def test_sailing_reward_logic():
    """Verifies angular difference reward calculation."""
    env = gym.make("Sailing-v0", grid_size=5)
    env.reset(seed=42) # wind starts at 1
    
    # Force wind to 1 for precise checks
    env.unwrapped.wind_dir = 1
    
    # Action 1 (DR) vs Wind 1 (DR) -> diff 0 -> reward 0
    _, r, _, _, _ = env.step(1)
    assert r == 0.0
    
    # Reset wind for next step check
    env.unwrapped.wind_dir = 1
    # Action 5 (UL) vs Wind 1 (DR) -> diff 4 -> reward -2.0
    _, r, _, _, _ = env.step(5)
    assert r == -2.0
    
    env.unwrapped.wind_dir = 1
    # Action 3 (DL) vs Wind 1 (DR) -> diff 2 -> reward -1.0
    _, r, _, _, _ = env.step(3)
    assert r == -1.0

def test_sailing_noop():
    """Verifies that moving out of bounds is a no-op."""
    env = gym.make("Sailing-v0", grid_size=5)
    env.reset() # Start at 0,0
    
    # At 0,0. Move LEFT (Action 4)
    obs, _, _, _, _ = env.step(4)
    assert obs == 0, "Moving left from 0,0 should be a no-op"
    
    # Move UP (Action 6)
    obs, _, _, _, _ = env.step(6)
    assert obs == 0, "Moving up from 0,0 should be a no-op"
    
    # Move to top-right boundary
    env.unwrapped.x = 4
    env.unwrapped.y = 0
    # Move RIGHT (Action 0)
    obs, _, _, _, _ = env.step(0)
    assert env.unwrapped.x == 4 and env.unwrapped.y == 0
