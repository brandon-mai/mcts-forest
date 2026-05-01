import pytest
import numpy as np
from mcts_forest.envs.gym_adapter import GymAdapter
from mcts_forest.core.uct import UCT
from mcts_forest.utils.registry import REGISTRY

@pytest.fixture(scope="function")
def deterministic_frozen_lake():
    """Returns a deterministic 4x4 FrozenLake environment."""
    custom_map = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    return GymAdapter("FrozenLake-v1", desc=custom_map, is_slippery=False)

@pytest.fixture(scope="function")
def stochastic_frozen_lake():
    """Returns a stochastic (slippery) 4x4 FrozenLake environment."""
    return GymAdapter("FrozenLake-v1", map_name="4x4", is_slippery=True)

@pytest.fixture(scope="function")
def uct_solver(deterministic_frozen_lake):
    """Returns a UCT solver pre-configured for the deterministic environment."""
    return UCT(deterministic_frozen_lake, simulation_limit=100, c=1.41)
