import jax
import jax.numpy as jnp
import mctx
from typing import Any

def random_rollout(rng_key, initial_state, initial_obs, action_mask_fn, env_step, reward_norm_fn, num_actions, max_depth=100):
    def cond_fn(carry):
        state, obs, key, total_reward, active, steps = carry
        return active & (steps < max_depth)

    def body_fn(carry):
        state, obs, key, total_reward, active, steps = carry
        mask = action_mask_fn(obs)
        probs = mask.astype(jnp.float32)
        sum_probs = jnp.sum(probs)
        probs = jnp.where(sum_probs > 0, probs / sum_probs, jnp.ones(num_actions) / float(num_actions))
        
        key, choice_key, step_key = jax.random.split(key, 3)
        action = jax.random.choice(choice_key, num_actions, p=probs)
        next_state, next_obs, reward, done, _ = env_step(step_key, state, action)
        
        norm_r = reward_norm_fn(reward)
        return next_state, next_obs, key, total_reward + norm_r * active, active & ~done, steps + 1

    # Initial carry: state, obs, key, total_reward, active, steps
    _, _, _, final_reward, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (initial_state, initial_obs, rng_key, 0.0, True, 0)
    )
    return final_reward

def jax_mctx_search(
    key,                    # Random key
    obs,                    # Observation
    state,                  # State
    env_step,               # Environment step function
    env_reset,              # Environment reset function
    action_mask_fn,         # Action mask function (input: obs)
    reward_norm_fn,         # Reward normalization function
    state_equal_fn,         # State equality function
    num_actions,            # Number of actions
    num_simulations=100,    # Number of simulations
    max_depth=10,           # Maximum depth
    gamma=0.99,             # Discount factor
):
    # Since solver_fn is vmapped by benchmax, obs and state represent a single environment state.
    # Expand to batch size 1 for mctx
    batched_obs = jax.tree.map(lambda x: jnp.expand_dims(x, 0), obs)
    batched_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
    
    # Root logits with masked invalid actions
    root_mask = action_mask_fn(obs)
    root_logits = jnp.where(root_mask, 0.0, -1e9)[None, :]
    
    # Value estimation for the root node
    key, rollout_key, search_key = jax.random.split(key, 3)
    root_value = random_rollout(
        rollout_key, state, obs,
        action_mask_fn, env_step, reward_norm_fn, num_actions, max_depth=100
    )
    root_value_batched = jnp.array([root_value])
    
    root = mctx.RootFnOutput(
        prior_logits=root_logits,
        value=root_value_batched,
        embedding=(batched_state, batched_obs)
    )
    
    def recurrent_fn(params, rng_key, action, embedding):
        state, obs = embedding
        
        def single_step(key, s, o, a):
            key_step, key_rollout = jax.random.split(key)
            next_s, next_o, reward, done, _ = env_step(key_step, s, a)
            
            estimated_value = random_rollout(
                key_rollout, next_s, next_o,
                action_mask_fn, env_step, reward_norm_fn, num_actions, max_depth=100
            )
            
            discount = jnp.where(done, 0.0, gamma)
            mask = action_mask_fn(next_o)
            prior_logits = jnp.where(mask, 0.0, -1e9)
            norm_reward = reward_norm_fn(reward)
            val = jnp.where(done, 0.0, estimated_value)
            
            return (next_s, next_o), norm_reward, discount, prior_logits, val

        batch_size = action.shape[0]
        keys = jax.random.split(rng_key, batch_size)
        vmapped_fn = jax.vmap(single_step, in_axes=(0, 0, 0, 0))
        next_embedding, rewards, discounts, prior_logits, values = vmapped_fn(keys, state, obs, action)
        
        return mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discounts,
            prior_logits=prior_logits,
            value=values
        ), next_embedding

    policy_output = mctx.gumbel_muzero_policy(
        params=None,
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        qtransform=mctx.qtransform_by_parent_and_siblings
    )
    
    return policy_output.action[0]
