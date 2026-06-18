import os
import numpy as np
from typing import Dict, Type, Any, Callable
from functools import lru_cache
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from mcts_forest.envs.gym_adapter import GymAdapter
from mcts_forest.core.gsp_uct import GSPUCT
from mcts_forest.core.gsp_uct_f import GSPUCTFull
from mcts_forest.core.gbopd import GBOPD
from mcts_forest.core.gbop import GBOP
from mcts_forest.core.random_agent import RandomAgent
from mcts_forest.envs.jair_envs import FactoredRiverSwimEnv, FourRoomsEnv, PassengerGridEnv, SysadminRingEnv

class Registry:
    def __init__(self):
        self.envs = {}
        self.solvers = {}

    def register_env(self, name, factory): self.envs[name.lower()] = factory
    def register_solver(self, name, factory): self.solvers[name.lower()] = factory
    def get_env(self, name, **kwargs):
        import re
        name_lower = name.lower()
        
        # 1. Handle probability suffixes like _0.8
        prob_match = re.search(r'_(\d+\.\d+)$', name_lower)
        if prob_match:
            prob = float(prob_match.group(1))
            base_name = name_lower[:prob_match.start()]
            
            # Special case for sailing: extract size if present (e.g., sailing10x10_0.8)
            sailing_match = re.match(r'sailing(\d+)x(\d+)', base_name)
            if sailing_match:
                kwargs["grid_size"] = int(sailing_match.group(1))
                base_name = "sailing"
            
            if base_name in self.envs:
                if "taxi_rain" in base_name:
                    kwargs["rainy_probability"] = prob
                elif "frozenlake_slip" in base_name:
                    kwargs["success_rate"] = prob 
                elif base_name == "sailing":
                    kwargs["wind_keep_probability"] = prob
                return self.envs[base_name](**kwargs)

        # 2. Handle size-only sailing (e.g., sailing10x10)
        sailing_match = re.match(r'sailing(\d+)x(\d+)', name_lower)
        if sailing_match:
            kwargs["grid_size"] = int(sailing_match.group(1))
            return self.envs["sailing"](**kwargs)
            
        # 3. Handle JAIR environments
        river_match = re.match(r'riverswim_n(\d+)x(\d+)', name_lower)
        if river_match:
            kwargs["num_rivers"] = int(river_match.group(1))
            kwargs["num_locations"] = int(river_match.group(2))
            return self.envs["riverswim"](**kwargs)
            
        fourrooms_match = re.match(r'fourrooms_n(\d+)', name_lower)
        if fourrooms_match:
            kwargs["n"] = int(fourrooms_match.group(1))
            return self.envs["fourrooms"](**kwargs)
            
        sysadmin_match = re.match(r'sysadmin_n(\d+)', name_lower)
        if sysadmin_match:
            kwargs["num_computers"] = int(sysadmin_match.group(1))
            return self.envs["sysadmin"](**kwargs)
            
        return self.envs[name_lower](**kwargs)
    def get_solver(self, name, env, **kwargs): return self.solvers[name.lower()](env, **kwargs)

REGISTRY = Registry()

# 1. Register Environments
@lru_cache(maxsize=1024)
def get_map(size=4, seed=None):
    return generate_random_map(size=size, seed=seed)

REGISTRY.register_env("frozenlake", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(4), is_slippery=False, **kwargs))
REGISTRY.register_env("frozenlake_slip", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(4), is_slippery=True, **kwargs))
REGISTRY.register_env("frozenlake8x8", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(8), is_slippery=False, **kwargs))
REGISTRY.register_env("frozenlake8x8_slip", lambda **kwargs: GymAdapter("FrozenLake-v1", desc=get_map(8), is_slippery=True, **kwargs))
REGISTRY.register_env("taxi", lambda **kwargs: GymAdapter("Taxi-v4", **kwargs))
REGISTRY.register_env("taxi_rain", lambda **kwargs: GymAdapter("Taxi-v4", is_rainy=True, **kwargs))
REGISTRY.register_env("cartpole", lambda **kwargs: GymAdapter("CartPole-v1", **kwargs))
REGISTRY.register_env("sailing", lambda **kwargs: GymAdapter("Sailing-v0", **kwargs))
REGISTRY.register_env("riverswim", lambda **kwargs: GymAdapter("FactoredRiverSwim-v0", **kwargs))
REGISTRY.register_env("fourrooms", lambda **kwargs: GymAdapter("FourRooms-v0", **kwargs))
REGISTRY.register_env("passenger_grid", lambda **kwargs: GymAdapter("PassengerGrid-v0", **kwargs))
REGISTRY.register_env("sysadmin", lambda **kwargs: GymAdapter("SysadminRing-v0", **kwargs))

# 2. Register Solvers
REGISTRY.register_solver("gsp_uct", lambda env, **kwargs: GSPUCT(env, **kwargs))
REGISTRY.register_solver("gsp_uct_f", lambda env, **kwargs: GSPUCTFull(env, **kwargs))
REGISTRY.register_solver("gbopd", lambda env, **kwargs: GBOPD(env, **kwargs))
REGISTRY.register_solver("gbop", lambda env, **kwargs: GBOP(env, **kwargs))

REGISTRY.register_solver("random", lambda env, **kwargs: RandomAgent(env, **kwargs))
