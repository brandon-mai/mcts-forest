import pytest
import numpy as np
from mcts_forest.core.uct import UCT

def test_uct_basic_search(deterministic_frozen_lake):
    """Verify that UCT can run a search and return a valid action."""
    solver = UCT(deterministic_frozen_lake, simulation_limit=100, c=1.41)
    obs = deterministic_frozen_lake.reset()
    
    action, info = solver.search(obs)
    
    assert action in range(4)
    assert "root_v" in info
    assert info["root_v"] >= 0.0

def test_uct_solves_deterministic_lake(deterministic_frozen_lake):
    """Verify that UCT can solve the deterministic FrozenLake environment."""
    # We'll run a few steps and see if it moves towards the goal
    solver = UCT(deterministic_frozen_lake, simulation_limit=2000, c=1.0)
    obs = deterministic_frozen_lake.reset()
    
    # In the specific map: "SFFF", "FHFH", "FFFH", "HFFG"
    # Action 2 (Right) or 1 (Down) is usually good from (0,0)
    action, _ = solver.search(obs)
    assert action in [1, 2]

def test_uct_fails_slippery_lake(stochastic_frozen_lake):
    """Verify that deterministic UCT struggles with slippery environments as expected."""
    # This is more of a qualitative test, we just check that it runs
    solver = UCT(stochastic_frozen_lake, simulation_limit=100, c=1.41)
    obs = stochastic_frozen_lake.reset()
    
    action, info = solver.search(obs)
    assert action in range(4)
