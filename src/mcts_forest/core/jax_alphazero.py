import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import numpy as np
from typing import NamedTuple, Any, Callable, Dict, Tuple, Optional
from functools import partial
from mcts_forest.core.jax_mctxf import jax_mctx_search

class ReplayBuffer(NamedTuple):
    obs: Any  # PyTree of buffer arrays
    target_policy: jnp.ndarray
    target_value: jnp.ndarray
    active_mask: jnp.ndarray
    pointer: jnp.ndarray
    current_size: jnp.ndarray
    buffer_size: int

def init_replay_buffer(buffer_size: int, obs_example: Any, num_actions: int) -> ReplayBuffer:
    obs_buffer = jax.tree.map(
        lambda x: jnp.zeros((buffer_size,) + x.shape, dtype=x.dtype),
        obs_example
    )
    return ReplayBuffer(
        obs=obs_buffer,
        target_policy=jnp.zeros((buffer_size, num_actions), dtype=jnp.float32),
        target_value=jnp.zeros((buffer_size,), dtype=jnp.float32),
        active_mask=jnp.zeros((buffer_size,), dtype=jnp.float32),
        pointer=jnp.array(0, dtype=jnp.int32),
        current_size=jnp.array(0, dtype=jnp.int32),
        buffer_size=buffer_size
    )

def add_to_buffer(buffer: ReplayBuffer, obs: Any, target_policy: jnp.ndarray, target_value: jnp.ndarray, active_mask: jnp.ndarray) -> ReplayBuffer:
    # Flatten shapes: input can be [Steps, B, ...] or [B, ...]
    # We flatten first dimension to batch transitions
    flat_obs = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), obs)
    flat_policy = target_policy.reshape((-1, target_policy.shape[-1]))
    flat_value = target_value.reshape((-1,))
    flat_mask = active_mask.reshape((-1,))
    
    n_transitions = flat_value.shape[0]
    indices = (buffer.pointer + jnp.arange(n_transitions)) % buffer.buffer_size
    
    new_obs = jax.tree.map(
        lambda buf_arr, batch_arr: buf_arr.at[indices].set(batch_arr),
        buffer.obs,
        flat_obs
    )
    new_target_policy = buffer.target_policy.at[indices].set(flat_policy)
    new_target_value = buffer.target_value.at[indices].set(flat_value)
    new_active_mask = buffer.active_mask.at[indices].set(flat_mask)
    
    new_pointer = (buffer.pointer + n_transitions) % buffer.buffer_size
    new_size = jnp.minimum(buffer.current_size + n_transitions, buffer.buffer_size)
    
    return buffer._replace(
        obs=new_obs,
        target_policy=new_target_policy,
        target_value=new_target_value,
        active_mask=new_active_mask,
        pointer=new_pointer,
        current_size=new_size
    )

def sample_buffer(key: jax.Array, buffer: ReplayBuffer, batch_size: int) -> Dict[str, Any]:
    idx = jax.random.randint(key, (batch_size,), 0, buffer.current_size)
    sampled_obs = jax.tree.map(lambda x: x[idx], buffer.obs)
    return {
        "obs": sampled_obs,
        "target_policy": buffer.target_policy[idx],
        "target_value": buffer.target_value[idx],
        "active_mask": buffer.active_mask[idx]
    }

def compute_returns(rewards: jnp.ndarray, discounts: jnp.ndarray, final_value: jnp.ndarray) -> jnp.ndarray:
    def d_step(suffix_return, step_inputs):
        r, d = step_inputs
        target_v = r + d * suffix_return
        return target_v, target_v

    _, values = jax.lax.scan(d_step, final_value, (rewards, discounts), reverse=True)
    return values

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
def self_play_episode_batch(
    key: jax.Array,
    nn_params: Any,
    nn_model: nn.Module,
    reset_fn: Callable,
    step_fn: Callable,
    action_mask_fn: Optional[Callable],
    reward_norm_fn: Optional[Callable],
    state_equal_fn: Optional[Callable],
    num_actions: int,
    num_parallel: int,
    max_episode_steps: int,
    num_simulations: int,
    max_depth: Optional[int],
    gamma: float,
    ucb_mode: str = "spuct",
    merge_mode: str = "depth_dependent",
    p: Any = 1.0,
    c: Any = 1.25
) -> Dict[str, Any]:
    init_key, loop_key = jax.random.split(key)
    seed_keys = jax.random.split(init_key, num_parallel)
    
    states, obss = jax.vmap(reset_fn)(seed_keys)
    active_mask = jnp.ones(num_parallel, dtype=jnp.bool_)
    
    carry = (states, obss, loop_key, active_mask)
    
    def scan_step(carry, _):
        states, obss, key, active_mask = carry
        
        key, search_key, env_key = jax.random.split(key, 3)
        search_keys = jax.random.split(search_key, num_parallel)
        env_keys = jax.random.split(env_key, num_parallel)
        
        # Parallel solver search with return_weights=True
        vmapped_search = jax.vmap(
            partial(
                jax_mctx_search,
                env_step=step_fn,
                env_reset=reset_fn,
                action_mask_fn=action_mask_fn,
                reward_norm_fn=reward_norm_fn,
                state_equal_fn=state_equal_fn,
                num_actions=num_actions,
                num_simulations=num_simulations,
                max_depth=max_depth,
                gamma=gamma,
                nn_model=nn_model,
                nn_params=nn_params,
                return_weights=True,
                ucb_mode=ucb_mode,
                merge_mode=merge_mode,
                p=p,
                c=c
            ),
            in_axes=(0, 0, 0)
        )
        actions, target_policies = vmapped_search(search_keys, obss, states)
        
        # Parallel env step
        next_states, next_obss, rewards, dones, _ = jax.vmap(step_fn)(env_keys, states, actions)
        
        # Build step info
        step_active = active_mask.astype(jnp.float32)
        step_discount = jnp.where(dones, 0.0, gamma) * step_active
        
        step_reward = rewards * step_active
        
        next_active_mask = active_mask & ~dones
        next_carry = (next_states, next_obss, key, next_active_mask)
        
        transition = {
            "obs": obss,
            "target_policy": target_policies,
            "reward": step_reward,
            "discount": step_discount,
            "active_mask": step_active
        }
        return next_carry, transition

    final_carry, trajectories = jax.lax.scan(scan_step, carry, xs=None, length=max_episode_steps)
    _, final_obss, _, _ = final_carry
    
    # Calculate undiscounted empirical returns in raw scale
    final_value = jnp.zeros(num_parallel)
    raw_returns = compute_returns(trajectories["reward"], trajectories["discount"], final_value)
    
    # Apply normalization to the final accumulated return
    if reward_norm_fn is not None:
        target_values = reward_norm_fn(raw_returns)
    else:
        target_values = raw_returns
    
    return {
        "obs": trajectories["obs"],
        "target_policy": trajectories["target_policy"],
        "target_value": target_values,
        "active_mask": trajectories["active_mask"],
        "reward": trajectories["reward"]
    }

def loss_fn(params: Any, model: nn.Module, batch: Dict[str, Any], l2_wd: float = 1e-4) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    pred_logits, pred_value = model.apply(params, batch["obs"])
    
    # policy cross entropy
    policy_loss = optax.softmax_cross_entropy(pred_logits, batch["target_policy"])
    policy_loss = (policy_loss * batch["active_mask"]).mean()
    
    # value MSE
    value_loss = jnp.square(pred_value - batch["target_value"])
    value_loss = (value_loss * batch["active_mask"]).mean()
    
    # L2 regularization
    l2_loss = sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(params))
    
    total_loss = policy_loss + value_loss + l2_wd * l2_loss
    
    metrics = {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "l2_loss": l2_loss
    }
    return total_loss, metrics

@partial(jax.jit, static_argnums=(2, 3))
def train_step(
    params: Any,
    opt_state: optax.OptState,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, Any]
) -> Tuple[Any, optax.OptState, Dict[str, jnp.ndarray]]:
    grad_fn = jax.value_and_grad(lambda p: loss_fn(p, model, batch), has_aux=True)
    (loss, metrics), grads = grad_fn(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Compute gradient norm
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    metrics["grad_norm"] = grad_norm
    
    return new_params, new_opt_state, metrics

@partial(jax.jit, static_argnums=(2, 3))
def train_steps_jit(
    params: Any,
    opt_state: optax.OptState,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    stacked_batch: Dict[str, Any]
) -> Tuple[Any, optax.OptState, Dict[str, jnp.ndarray]]:
    def scan_fn(carry, batch):
        p, opt_s = carry
        p, opt_s, metrics = train_step(p, opt_s, model, optimizer, batch)
        return (p, opt_s), metrics

    (new_params, new_opt_state), metrics_history = jax.lax.scan(scan_fn, (params, opt_state), stacked_batch)
    return new_params, new_opt_state, metrics_history

class CpuReplayBuffer:
    def __init__(self, buffer_size: int, obs_example: Any, num_actions: int):
        self.buffer_size = buffer_size
        self.pointer = 0
        self.current_size = 0
        
        self.obs = jax.tree.map(
            lambda x: np.zeros((buffer_size,) + x.shape, dtype=np.asarray(x).dtype),
            obs_example
        )
        self.target_policy = np.zeros((buffer_size, num_actions), dtype=np.float32)
        self.target_value = np.zeros((buffer_size,), dtype=np.float32)
        self.active_mask = np.zeros((buffer_size,), dtype=np.float32)
        
    def add(self, obs: Any, target_policy: jnp.ndarray, target_value: jnp.ndarray, active_mask: jnp.ndarray):
        # Convert JAX device arrays to NumPy on the host (non-blocking if async, but runs in background thread anyway)
        obs_np = jax.tree.map(np.asarray, obs)
        policy_np = np.asarray(target_policy)
        value_np = np.asarray(target_value)
        mask_np = np.asarray(active_mask)
        
        flat_obs = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), obs_np)
        flat_policy = policy_np.reshape((-1, policy_np.shape[-1]))
        flat_value = value_np.reshape((-1,))
        flat_mask = mask_np.reshape((-1,))
        
        n_transitions = flat_value.shape[0]
        indices = (self.pointer + np.arange(n_transitions)) % self.buffer_size
        
        jax.tree.map(
            lambda buf_arr, batch_arr: buf_arr.at[indices].set(batch_arr) if hasattr(buf_arr, 'at') else (buf_arr.__setitem__(indices, batch_arr)),
            self.obs,
            flat_obs
        )
        
        # NumPy array assignment
        self.target_policy[indices] = flat_policy
        self.target_value[indices] = flat_value
        self.active_mask[indices] = flat_mask
        
        self.pointer = (self.pointer + n_transitions) % self.buffer_size
        self.current_size = min(self.current_size + n_transitions, self.buffer_size)
        
    def sample(self, key_or_rng: np.random.Generator, batch_size: int) -> Dict[str, Any]:
        idx = key_or_rng.integers(0, self.current_size, size=batch_size)
        sampled_obs = jax.tree.map(lambda x: x[idx], self.obs)
        return {
            "obs": sampled_obs,
            "target_policy": self.target_policy[idx],
            "target_value": self.target_value[idx],
            "active_mask": self.active_mask[idx]
        }
