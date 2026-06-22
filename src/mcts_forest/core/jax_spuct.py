import jax
import jax.numpy as jnp
from typing import Any
from functools import partial

def _rollout_step(carry, _, action_mask_fn, env_step, reward_norm_fn, num_actions):
    state, obs, key, total_r, active = carry
    mask = action_mask_fn(obs)
    probs = mask.astype(jnp.float32)
    sum_probs = jnp.sum(probs)
    probs = jnp.where(sum_probs > 0, probs / sum_probs, jnp.ones(num_actions) / float(num_actions))
    key, choice_key, step_key = jax.random.split(key, 3)
    action = jax.random.choice(choice_key, num_actions, p=probs)
    next_state, next_obs, reward, done, _ = env_step(step_key, state, action)
    norm_r = reward_norm_fn(reward)
    return (next_state, next_obs, key, total_r + norm_r * active, active & ~done), None

def _rollout_fn(key, start_state, start_obs, rollout_depth, rollout_step_fn):
    (final_state, final_obs, final_key, reward_sum, _), _ = jax.lax.scan(
        rollout_step_fn, (start_state, start_obs, key, 0.0, True), None, length=rollout_depth
    )
    return reward_sum

def _forward_step(
    f_carry, step_idx,
    action_mask_fn, env_step, state_equal_fn,
    c, num_actions, merge_mode, ucb_mode
):
    (
        u, curr_state, curr_obs, key, expanded_this_sim, is_done, next_empty_idx,
        t_states, t_valid, t_depths, t_V, t_T_s, t_Q, t_T_sa,
        carry_expanded_state, carry_expanded_obs
    ) = f_carry
    
    # Condition to take a real step
    take_step = ~(is_done | expanded_this_sim)
    
    # Select action
    mask = action_mask_fn(curr_obs)
    if ucb_mode == "standard":
        ucb = t_Q[u] + c * jnp.sqrt(t_T_s[u] + 1e-8) / (t_T_sa[u] + 1e-5)
    else: # spuct
        ucb = t_Q[u] + c * jnp.power(t_T_s[u] + 1.0, 0.25) / (jnp.sqrt(t_T_sa[u]) + 1e-5)
        
    key, noise_key = jax.random.split(key)
    noise = jax.random.uniform(noise_key, shape=ucb.shape) * 1e-6
    ucb = jnp.where(mask, ucb + noise, -1e9)
    action = jnp.argmax(ucb)
    
    # Environment step
    key, step_key = jax.random.split(key)
    next_state, next_obs, reward, done, _ = env_step(step_key, curr_state, action)
    
    # Check if next_state is already in tree
    equals = jax.vmap(state_equal_fn, in_axes=(0, None))(t_states, next_state)
    
    if merge_mode == "pure_tree":
        matches = jnp.zeros_like(t_valid)
    elif merge_mode == "depth_dependent":
        matches = equals & t_valid & (t_depths == (step_idx + 1))
    else: # depth_independent
        matches = equals & t_valid
        
    found = jnp.any(matches)
    found_idx = jnp.argmax(matches)
    
    # Calculate next node index
    v = jnp.where(found, found_idx, next_empty_idx)
    
    # Perform expansion if not found and taking step
    should_expand = take_step & ~found
    
    # Update tree arrays on expansion
    new_t_valid = t_valid.at[v].set(jnp.where(should_expand, True, t_valid[v]))
    new_t_depths = t_depths.at[v].set(jnp.where(should_expand, step_idx + 1, t_depths[v]))
    
    # Avoid lax.cond to prevent lambda re-creation / trace cache misses
    new_t_states = jax.tree.map(
        lambda x, y: x.at[v].set(jnp.where(should_expand, y, x[v])),
        t_states,
        next_state
    )
    
    # Track newly expanded state and observation
    new_expanded_state = jax.tree.map(lambda x, y: jnp.where(should_expand, y, x), carry_expanded_state, next_state)
    new_expanded_obs = jax.tree.map(lambda x, y: jnp.where(should_expand, y, x), carry_expanded_obs, next_obs)
    
    # If we expanded, we update flag and empty index
    new_expanded = expanded_this_sim | should_expand
    new_next_empty_idx = next_empty_idx + jnp.where(should_expand, 1, 0)
    
    # Trajectory tracking
    step_info = (take_step, u, action, reward, done, v)
    
    # Next iteration carry
    next_u = jnp.where(take_step, v, u)
    next_carry_state = jax.tree.map(lambda x, y: jnp.where(take_step, y, x), curr_state, next_state)
    next_carry_obs = jax.tree.map(lambda x, y: jnp.where(take_step, y, x), curr_obs, next_obs)
    
    next_f_carry = (
        next_u, next_carry_state, next_carry_obs, key, new_expanded, is_done | (take_step & done),
        new_next_empty_idx, new_t_states, new_t_valid, new_t_depths, t_V, t_T_s, t_Q, t_T_sa,
        new_expanded_state, new_expanded_obs
    )
    return next_f_carry, step_info

def _backward_step(b_carry, step_data, reward_norm_fn, gamma, p):
    t_V, t_T_s, t_Q, t_T_sa = b_carry
    active, u, a, r, done, v = step_data
    
    norm_r = reward_norm_fn(r)
    V_next = jnp.where(done, 0.0, t_V[v])
    
    # Update Q
    old_Q = t_Q[u, a]
    old_T_sa = t_T_sa[u, a]
    new_Q = (old_Q * old_T_sa + norm_r + gamma * V_next) / (old_T_sa + 1.0)
    
    updated_Q = t_Q.at[u, a].set(jnp.where(active, new_Q, old_Q))
    updated_T_sa = t_T_sa.at[u, a].set(jnp.where(active, old_T_sa + 1.0, old_T_sa))
    
    # Update V and T_s
    updated_T_s = t_T_s.at[u].set(jnp.where(active, t_T_s[u] + 1.0, t_T_s[u]))
    
    # V update formula: V = (sum_a ((T_sa / T_s) * (Q ** p))) ** (1/p)
    sum_val = jnp.sum((updated_T_sa[u] / (updated_T_s[u] + 1e-8)) * jnp.power(jnp.maximum(updated_Q[u], 0.0), p))
    new_V = jnp.power(sum_val, 1.0 / p)
    
    updated_V = t_V.at[u].set(jnp.where(active, new_V, t_V[u]))
    
    return (updated_V, updated_T_s, updated_Q, updated_T_sa), None

def _simulation_step(
    carry, _,
    state, obs, max_depth, rollout_fn,
    forward_step_fn, backward_step_fn
):
    tree_states, tree_valid, tree_depths, tree_V, tree_T_s, tree_Q, tree_T_sa, tree_next_empty_idx, sim_key = carry
    
    # 1. Forward Pass (Selection & Expansion)
    init_f_carry = (
        0, state, obs, sim_key, False, False, tree_next_empty_idx,
        tree_states, tree_valid, tree_depths, tree_V, tree_T_s, tree_Q, tree_T_sa,
        state, obs
    )
    
    final_f_carry, trajectory = jax.lax.scan(forward_step_fn, init_f_carry, jnp.arange(max_depth))
    
    (
        _, _, _, post_key, _, _, new_next_empty_idx,
        new_states, new_valid, new_depths, new_V, _, _, _,
        final_expanded_state, final_expanded_obs
    ) = final_f_carry
    
    # Leaf evaluation (rollout) - run exactly once per simulation
    post_key, rollout_key = jax.random.split(post_key)
    leaf_v = rollout_fn(rollout_key, final_expanded_state, final_expanded_obs)
    
    expanded_any = new_next_empty_idx > tree_next_empty_idx
    expanded_idx = tree_next_empty_idx
    new_V = new_V.at[expanded_idx].set(jnp.where(expanded_any, leaf_v, new_V[expanded_idx]))
    
    path_active, path_u, path_a, path_r, path_done, path_v = trajectory
    
    # 2. Backward Pass (Backpropagation)
    # Scan backward over trajectory to update Q and V values
    rev_active = jnp.flip(path_active)
    rev_u = jnp.flip(path_u)
    rev_a = jnp.flip(path_a)
    rev_r = jnp.flip(path_r)
    rev_done = jnp.flip(path_done)
    rev_v = jnp.flip(path_v)
    
    init_b_carry = (new_V, tree_T_s, tree_Q, tree_T_sa)
    final_b_carry, _ = jax.lax.scan(
        backward_step_fn,
        init_b_carry,
        (rev_active, rev_u, rev_a, rev_r, rev_done, rev_v)
    )
    
    final_V, final_T_s, final_Q, final_T_sa = final_b_carry
    
    next_carry = (
        new_states, new_valid, new_depths, final_V, final_T_s, final_Q, final_T_sa, new_next_empty_idx, post_key
    )
    return next_carry, None

def jax_spuct_search(
    key,
    obs,
    state,
    env_step,
    env_reset,
    action_mask_fn,
    reward_norm_fn,
    state_equal_fn,
    num_actions,
    num_simulations=100,
    max_depth=10,
    gamma=0.99,
    c=1.414,
    p=1.0,
    rollout_depth=500,
    merge_mode="depth_independent",
    horizon_mode="fixed",
    ucb_mode="spuct"
):
    """
    JAX-native Stochastic Power UCT (SP-UCT) solver.
    """
    if horizon_mode == "adaptive":
        import math
        if gamma >= 1.0:
            max_depth = 10
        else:
            max_depth = int(math.ceil(math.log(num_simulations) / (2.0 * math.log(1.0 / gamma))))
    elif horizon_mode == "infinite":
        max_depth = 500

    max_nodes = num_simulations + 2

    # Initialize tree structures
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (max_nodes,) + x.shape), state)
    state_valid = jnp.zeros(max_nodes, dtype=jnp.bool_).at[0].set(True)
    state_depths = jnp.zeros(max_nodes, dtype=jnp.int32)
    
    # Initialize rollout for root
    root_mask = action_mask_fn(obs)
    probs = root_mask.astype(jnp.float32)
    sum_probs = jnp.sum(probs)
    probs = jnp.where(sum_probs > 0, probs / sum_probs, jnp.ones(num_actions) / float(num_actions))
    key, rollout_key = jax.random.split(key)
    
    # Partially bound top-level functions to avoid inner redefinitions
    rollout_step_fn = partial(
        _rollout_step,
        action_mask_fn=action_mask_fn,
        env_step=env_step,
        reward_norm_fn=reward_norm_fn,
        num_actions=num_actions
    )
    
    rollout_fn = partial(
        _rollout_fn,
        rollout_depth=rollout_depth,
        rollout_step_fn=rollout_step_fn
    )
    
    forward_step_fn = partial(
        _forward_step,
        action_mask_fn=action_mask_fn,
        env_step=env_step,
        state_equal_fn=state_equal_fn,
        c=c,
        num_actions=num_actions,
        merge_mode=merge_mode,
        ucb_mode=ucb_mode
    )
    
    backward_step_fn = partial(
        _backward_step,
        reward_norm_fn=reward_norm_fn,
        gamma=gamma,
        p=p
    )
    
    simulation_step_fn = partial(
        _simulation_step,
        state=state,
        obs=obs,
        max_depth=max_depth,
        rollout_fn=rollout_fn,
        forward_step_fn=forward_step_fn,
        backward_step_fn=backward_step_fn
    )

    root_v = rollout_fn(rollout_key, state, obs)
    V = jnp.zeros(max_nodes, dtype=jnp.float32).at[0].set(root_v)
    T_s = jnp.zeros(max_nodes, dtype=jnp.float32)
    Q = jnp.zeros((max_nodes, num_actions), dtype=jnp.float32)
    T_sa = jnp.zeros((max_nodes, num_actions), dtype=jnp.float32)
    
    next_empty_idx = 1
    
    init_carry = (states, state_valid, state_depths, V, T_s, Q, T_sa, next_empty_idx, key)
    final_carry, _ = jax.lax.scan(simulation_step_fn, init_carry, None, length=num_simulations)
    
    # Select greedy action from root node (index 0) with tie-breaking noise
    final_Q_root = final_carry[5][0]
    mask = action_mask_fn(obs)
    
    key, tie_key = jax.random.split(final_carry[8])
    noise = jax.random.uniform(tie_key, shape=final_Q_root.shape) * 1e-6
    final_Q_root = jnp.where(mask, final_Q_root + noise, -1e9)
    action = jnp.argmax(final_Q_root)
    return action
