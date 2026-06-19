import jax
import jax.numpy as jnp
from typing import Any

def jax_random_search(key, obs, state, env_step, env_reset, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions):
    """
    A unified functional JAX-compatible random search.
    Accepts env_step and env_reset to future-proof for planning/MCTS.
    """
    # Retrieve valid actions mask
    mask = action_mask_fn(obs)
    
    # Calculate probabilities for valid actions
    probs = mask.astype(jnp.float32)
    sum_probs = jnp.sum(probs)
    
    # Normalize probabilities safely
    probs = jnp.where(sum_probs > 0, probs / sum_probs, jnp.ones(num_actions, dtype=jnp.float32) / float(num_actions))
    
    # Sample random action
    action = jax.random.choice(key, num_actions, p=probs)
    return action
