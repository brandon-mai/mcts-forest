import numpy as np
from typing import Tuple, Dict, Any

class RandomAgent:
    """A baseline agent that selects actions uniformly at random."""
    def __init__(self, env, **kwargs):
        self.env = env
        self.num_actions = getattr(env, 'action_space_size', 0)
        self.simulation_limit = kwargs.get('simulation_limit', 0)

    def search(self, initial_state: int, **kwargs) -> Tuple[int, Dict[str, Any]]:
        action = np.random.randint(0, self.num_actions)
        return action, {}

    def get_name(self) -> str:
        return "random"
