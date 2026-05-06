import pytest
import numpy as np
from mcts_forest.envs.jair_envs import FactoredRiverSwimEnv, FourRoomsEnv, PassengerGridEnv, SysadminRingEnv
from mcts_forest.utils.registry import REGISTRY

def test_riverswim():
    env = FactoredRiverSwimEnv(num_rivers=1, num_locations=4, time_limit=10)
    obs, _ = env.reset(seed=42)
    assert obs == 0 # (0 spatial, 0 time)
    
    # Move downstream from 0 should stay at 0
    next_obs, reward, term, trunc, _ = env.step(0)
    assert env.positions[0] == 0
    assert reward == 0.1 / 2.0 # START_REST_REWARD / (2*N)
    
    # Test state snapshot
    state = env.get_state()
    env.step(1)
    env.set_state(state)
    assert env.get_state() == state

def test_fourrooms():
    env = FourRoomsEnv(n=2, time_limit=10) # 5x5 grid
    obs, _ = env.reset(seed=42)
    assert env.grid_size == 5
    
    # Verify wall detection
    # Wall at x=2, y=2. Doors at (0, 2), (3, 2), (2, 0), (2, 3)
    env.doors = [10, 13, 2, 17]
    env.x, env.y = 1, 1
    env.time = 0
    # Move RIGHT (action 2) into wall at (2, 1)
    # Note: FourRooms is slippery by default (50% target), 
    # so we set slippery=False for deterministic test
    env.slippery = False
    env.step(2) 
    assert env.x == 1 # Blocked by wall at (2, 1)
    
    # Move DOWN (action 1) to (1, 1) -> (1, 2) which is a wall
    env.step(1)
    assert env.y == 1 # Blocked by wall at (1, 2)
    
    # Move LEFT (action 0) to (0, 1), then DOWN (action 1) to (0, 2) which is a DOOR
    env.x, env.y = 1, 1
    env.step(0) # (0, 1)
    env.step(1) # (0, 2) - Door
    assert env.y == 2

def test_passenger():
    env = PassengerGridEnv(time_limit=10, slippery=False)
    env.reset(seed=42)
    
    # First passenger at (1, 2)
    # Move: RIGHT, DOWN, DOWN
    env.step(2) # (1, 0)
    env.step(1) # (1, 1)
    env.step(1) # (1, 2)
    
    assert env.mask == 1 # First passenger picked up
    
def test_sysadmin():
    env = SysadminRingEnv(num_computers=5, time_limit=10)
    env.reset(seed=42)
    
    # Reboot computer 2
    _, reward, _, _, _ = env.step(2)
    assert (env.mask >> 2) & 1 == 1
    assert reward > 0

@pytest.mark.parametrize("env_name", [
    "riverswim_n2x4",
    "fourrooms_n3",
    "passenger_grid",
    "sysadmin_n5"
])
def test_registry_jair(env_name):
    env = REGISTRY.get_env(env_name)
    assert env is not None
    env.reset()
    env.step(0)
