import pytest
import numpy as np
from mcts_forest.core.uct import UCT
from mcts_forest.core.base import random_rollout_discrete

@pytest.mark.parametrize("seed", [42, 123, 2024])
def test_frozen_lake_deterministic_convergence(deterministic_frozen_lake, seed):
    """Verifies that UCT can solve deterministic Frozen Lake for various seeds."""
    adapter = deterministic_frozen_lake
    solver = UCT(adapter, simulation_limit=300)
    
    obs = adapter.reset(seed=seed)
    terminated, truncated = False, False
    total_reward = 0
    steps = 0
    
    while not (terminated or truncated) and steps < 20:
        action, _ = solver.search(obs)
        obs, reward, terminated, truncated, _ = adapter.step(action)
        total_reward += reward
        steps += 1
        
    assert total_reward == 1.0, f"Failed on seed {seed}: steps={steps}, reward={total_reward}"

def test_rollout_reward_range(deterministic_frozen_lake):
    """Verifies that random rollout reward falls within expected [0, 1] range for Frozen Lake."""
    dynamics = deterministic_frozen_lake.get_numba_dynamics()
    obs = deterministic_frozen_lake.reset()
    
    reward = random_rollout_discrete(obs, *dynamics, limit=100, gamma=0.99)
    assert 0.0 <= reward <= 1.0

def test_uct_info_consistency(deterministic_frozen_lake):
    """Verifies that search returns valid info dictionary."""
    adapter = deterministic_frozen_lake
    solver = UCT(adapter, simulation_limit=50)
    obs = adapter.reset()
    
    action, info = solver.search(obs)
    assert "root_v" in info
    assert isinstance(info["root_v"], float)
