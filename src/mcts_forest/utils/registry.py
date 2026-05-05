import os
import torch
import numpy as np
from typing import Dict, Type, Any, Callable
from functools import lru_cache
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from mcts_forest.envs.gym_adapter import GymAdapter
from mcts_forest.core.uct import UCT
from mcts_forest.core.stochastic_uct import StochasticUCT
from mcts_forest.core.sp_uct import SPUCT
from mcts_forest.core.openloop_mcts import OpenLoopMCTS
from mcts_forest.core.mcgs import MCGS
from mcts_forest.core.gsp_uct import GSPUCT
from mcts_forest.core.gsp_uct_f import GSPUCTFull
from mcts_forest.core.gbopd import GBOPD
from mcts_forest.core.gbop import GBOP
from mcts_forest.core.ments import MENTS
from mcts_forest.core.gsp_alphazero import GSPAlphaZero, GSPAlphaZeroNet
from mcts_forest.core.gsp_muzero import GSPMuZero, GSPMuZeroNet
from mcts_forest.core.gsp_stochastic_muzero import GSPStochasticMuZero, GSPStochasticMuZeroNet
from mcts_forest.core.alphazero import AlphaZero, AlphaZeroNet
from mcts_forest.core.muzero import MuZero, MuZeroNet
from mcts_forest.core.base import TorchModelAdapter
from mcts_forest.core.random_agent import RandomAgent

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

# 2. Register Solvers
REGISTRY.register_solver("uct", lambda env, **kwargs: UCT(env, **kwargs))
REGISTRY.register_solver("stochastic_uct", lambda env, **kwargs: StochasticUCT(env, **kwargs))
REGISTRY.register_solver("sp_uct", lambda env, **kwargs: SPUCT(env, **kwargs))
REGISTRY.register_solver("openloop_mcts", lambda env, **kwargs: OpenLoopMCTS(env, **kwargs))
REGISTRY.register_solver("mcgs", lambda env, **kwargs: MCGS(env, **kwargs))
REGISTRY.register_solver("gsp_uct", lambda env, **kwargs: GSPUCT(env, **kwargs))
REGISTRY.register_solver("gsp_uct_f", lambda env, **kwargs: GSPUCTFull(env, **kwargs))
REGISTRY.register_solver("gbopd", lambda env, **kwargs: GBOPD(env, **kwargs))
REGISTRY.register_solver("gbop", lambda env, **kwargs: GBOP(env, **kwargs))
REGISTRY.register_solver("ments", lambda env, **kwargs: MENTS(env, **kwargs))

def _get_zero_solver(env, solver_class, net_class, algo_name, **kwargs):
    model_path = f"checkpoints/{algo_name}/latest.pt"
    adapter = TorchModelAdapter(net_class())
    if os.path.exists(model_path):
        adapter.load(model_path)
    return solver_class(env, adapter, **kwargs)

REGISTRY.register_solver("gsp_alphazero", lambda env, **kwargs: _get_zero_solver(env, GSPAlphaZero, GSPAlphaZeroNet, "gsp_alphazero", **kwargs))
REGISTRY.register_solver("gsp_muzero", lambda env, **kwargs: _get_zero_solver(env, GSPMuZero, GSPMuZeroNet, "gsp_muzero", **kwargs))
REGISTRY.register_solver("gsp_stochastic_muzero", lambda env, **kwargs: _get_zero_solver(env, GSPStochasticMuZero, GSPStochasticMuZeroNet, "gsp_stochastic_muzero", **kwargs))
REGISTRY.register_solver("alphazero", lambda env, **kwargs: _get_zero_solver(env, AlphaZero, AlphaZeroNet, "alphazero", **kwargs))
REGISTRY.register_solver("muzero", lambda env, **kwargs: _get_zero_solver(env, MuZero, MuZeroNet, "muzero", **kwargs))
REGISTRY.register_solver("random", lambda env, **kwargs: RandomAgent(env, **kwargs))
