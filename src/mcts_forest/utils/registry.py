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
from mcts_forest.core.gsp_alphazero import GSPAlphaZero, GSPAlphaZeroNet
from mcts_forest.core.gsp_muzero import GSPMuZero, GSPMuZeroNet
from mcts_forest.core.gsp_stochastic_muzero import GSPStochasticMuZero, GSPStochasticMuZeroNet
from mcts_forest.core.base import TorchModelAdapter
from mcts_forest.core.random_agent import RandomAgent

class Registry:
    def __init__(self):
        self.envs = {}
        self.solvers = {}

    def register_env(self, name, factory): self.envs[name.lower()] = factory
    def register_solver(self, name, factory): self.solvers[name.lower()] = factory
    def get_env(self, name, **kwargs): return self.envs[name.lower()](**kwargs)
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
REGISTRY.register_env("taxi", lambda **kwargs: GymAdapter("Taxi-v3", **kwargs))
REGISTRY.register_env("taxi_rain", lambda **kwargs: GymAdapter("Taxi-v3", is_rainy=True, **kwargs))

# 2. Register Solvers
REGISTRY.register_solver("uct", lambda env, **kwargs: UCT(env, **kwargs))
REGISTRY.register_solver("stochastic_uct", lambda env, **kwargs: StochasticUCT(env, **kwargs))
REGISTRY.register_solver("sp_uct", lambda env, **kwargs: SPUCT(env, **kwargs))
REGISTRY.register_solver("openloop_mcts", lambda env, **kwargs: OpenLoopMCTS(env, **kwargs))
REGISTRY.register_solver("mcgs", lambda env, **kwargs: MCGS(env, **kwargs))
REGISTRY.register_solver("gsp_uct", lambda env, **kwargs: GSPUCT(env, **kwargs))
REGISTRY.register_solver("gsp_uct_f", lambda env, **kwargs: GSPUCTFull(env, **kwargs))

def _get_zero_solver(env, solver_class, net_class, algo_name, **kwargs):
    model_path = f"checkpoints/{algo_name}/latest.pt"
    adapter = TorchModelAdapter(net_class())
    if os.path.exists(model_path):
        adapter.load(model_path)
    return solver_class(env, adapter, **kwargs)

REGISTRY.register_solver("gsp_alphazero", lambda env, **kwargs: _get_zero_solver(env, GSPAlphaZero, GSPAlphaZeroNet, "gsp_alphazero", **kwargs))
REGISTRY.register_solver("gsp_muzero", lambda env, **kwargs: _get_zero_solver(env, GSPMuZero, GSPMuZeroNet, "gsp_muzero", **kwargs))
REGISTRY.register_solver("gsp_stochastic_muzero", lambda env, **kwargs: _get_zero_solver(env, GSPStochasticMuZero, GSPStochasticMuZeroNet, "gsp_stochastic_muzero", **kwargs))
REGISTRY.register_solver("random", lambda env, **kwargs: RandomAgent(env, **kwargs))
