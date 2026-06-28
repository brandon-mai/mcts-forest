import jax
import jax.numpy as jnp
import mctx
import chex
import functools
from typing import Optional, Any, Callable, Tuple, TypeVar, NamedTuple
from mctx._src import base
from mctx._src import tree as tree_lib
from mctx._src import action_selection
from mctx._src import qtransforms
from mctx._src import seq_halving

Tree = tree_lib.Tree
T = TypeVar("T")

# =====================================================================
# Core Recurrent & Rollout Functions (Customized)
# =====================================================================

def run_random_rollout(rng_key, initial_state, initial_obs, env_step, reward_norm_fn, num_actions, rollout_depth, action_mask_fn):
    """Executes a random rollout from a state up to rollout_depth using a JAX while_loop."""
    def cond_fn(loop_state):
        state, obs, key, done, depth, total_reward = loop_state
        return jnp.logical_and(jnp.logical_not(done), depth < rollout_depth)
        
    def body_fn(loop_state):
        state, obs, current_key, _, depth, total_reward = loop_state
        key, choice_key, step_key = jax.random.split(current_key, 3)
        
        mask = action_mask_fn(obs) if action_mask_fn is not None else jnp.ones(num_actions, dtype=jnp.bool_)
        probs = mask.astype(jnp.float32)
        sum_probs = jnp.sum(probs)
        probs = jnp.where(sum_probs > 0, probs / sum_probs, jnp.ones(num_actions) / float(num_actions))
        
        action = jax.random.choice(choice_key, num_actions, p=probs)
        next_state, next_obs, reward, done, _ = env_step(step_key, state, action)
        norm_r = reward_norm_fn(reward) if reward_norm_fn is not None else reward
        
        return next_state, next_obs, key, done, depth + 1, total_reward + norm_r

    init_done = False
    _, _, _, _, _, final_reward = jax.lax.while_loop(
        cond_fn, 
        body_fn, 
        (initial_state, initial_obs, rng_key, init_done, 0, 0.0)
    )
    return final_reward


# =====================================================================
# Injected MCTX Source Code (search.py, policies.py)
# =====================================================================

def _mask_invalid_actions(logits, invalid_actions):
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)


def _get_logits_from_probs(probs):
  tiny = jnp.finfo(probs.dtype).tiny
  return jnp.log(jnp.maximum(probs, tiny))


def _add_dirichlet_noise(rng_key, probs, *, dirichlet_alpha, dirichlet_fraction):
  chex.assert_rank(probs, 2)
  chex.assert_type([dirichlet_alpha, dirichlet_fraction], float)
  batch_size, num_actions = probs.shape
  noise = jax.random.dirichlet(
      rng_key,
      alpha=jnp.full([num_actions], fill_value=dirichlet_alpha),
      shape=(batch_size,))
  noisy_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * noise
  return noisy_probs


def _apply_temperature(logits, temperature):
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  tiny = jnp.finfo(logits.dtype).tiny
  return logits / jnp.maximum(tiny, temperature)


def _make_stochastic_recurrent_fn(
    decision_node_fn: base.DecisionRecurrentFn,
    chance_node_fn: base.ChanceRecurrentFn,
    num_actions: int,
    num_chance_outcomes: int,
) -> base.RecurrentFn:
  def stochastic_recurrent_fn(
      params: base.Params,
      rng: chex.PRNGKey,
      action_or_chance: base.Action,
      state: base.StochasticRecurrentState
  ) -> Tuple[base.RecurrentFnOutput, base.StochasticRecurrentState]:
    batch_size = jax.tree_util.tree_leaves(state.state_embedding)[0].shape[0]
    action = action_or_chance - 0
    chance_outcome = action_or_chance - num_actions

    decision_output, afterstate_embedding = decision_node_fn(
        params, rng, action, state.state_embedding)
    output_if_decision_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            jnp.full([batch_size, num_actions], fill_value=-jnp.inf),
            decision_output.chance_logits], axis=-1),
        value=decision_output.afterstate_value,
        reward=jnp.zeros_like(decision_output.afterstate_value),
        discount=jnp.ones_like(decision_output.afterstate_value))

    chance_output, state_embedding = chance_node_fn(params, rng, chance_outcome,
                                                    state.afterstate_embedding)
    output_if_chance_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            chance_output.action_logits,
            jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
            ], axis=-1),
        value=chance_output.value,
        reward=chance_output.reward,
        discount=chance_output.discount)

    new_state = base.StochasticRecurrentState(
        state_embedding=state_embedding,
        afterstate_embedding=afterstate_embedding,
        is_decision_node=jnp.logical_not(state.is_decision_node))

    def _broadcast_where(decision_leaf, chance_leaf):
      extra_dims = [1] * (len(decision_leaf.shape) - 1)
      expanded_is_decision = jnp.reshape(state.is_decision_node,
                                         [-1] + extra_dims)
      return jnp.where(expanded_is_decision, decision_leaf, chance_leaf)

    output = jax.tree.map(_broadcast_where, output_if_decision_node, output_if_chance_node)
    return output, new_state

  return stochastic_recurrent_fn


def _mask_tree(tree: Tree, num_actions: int, mode: str) -> Tree:
  def _take_slice(x):
    if mode == 'decision':
      return x[..., :num_actions]
    elif mode == 'chance':
      return x[..., num_actions:]
    else:
      raise ValueError(f'Unknown mode: {mode}.')

  return tree.replace(
      children_index=_take_slice(tree.children_index),
      children_prior_logits=_take_slice(tree.children_prior_logits),
      children_visits=_take_slice(tree.children_visits),
      children_rewards=_take_slice(tree.children_rewards),
      children_discounts=_take_slice(tree.children_discounts),
      children_values=_take_slice(tree.children_values),
      root_invalid_actions=_take_slice(tree.root_invalid_actions))


def _make_stochastic_action_selection_fn(
    decision_node_selection_fn: base.InteriorActionSelectionFn,
    num_actions: int,
) -> base.InteriorActionSelectionFn:
  def _chance_node_selection_fn(tree: Tree, node_index: chex.Array) -> chex.Array:
    num_chance = tree.children_visits[node_index]
    chance_logits = tree.children_prior_logits[node_index]
    prob_chance = jax.nn.softmax(chance_logits)
    argmax_chance = jnp.argmax(prob_chance / (num_chance + 1), axis=-1).astype(jnp.int32)
    return argmax_chance

  def _action_selection_fn(key: chex.PRNGKey, tree: Tree, node_index: chex.Array, depth: chex.Array) -> chex.Array:
    is_decision = tree.embeddings.is_decision_node[node_index]
    chance_selection = _chance_node_selection_fn(
        tree=_mask_tree(tree, num_actions, 'chance'),
        node_index=node_index) + num_actions
    decision_selection = decision_node_selection_fn(
        key, _mask_tree(tree, num_actions, 'decision'), node_index, depth)
    return jax.lax.cond(is_decision, lambda: decision_selection, lambda: chance_selection)

  return _action_selection_fn


# =====================================================================
# Injected Search Logic (search.py)
# =====================================================================

def update(x, vals, *indices):
  return x.at[indices].set(vals)

batch_update = jax.vmap(update)

def update_tree_node(tree: Tree[T], node_index: chex.Array, prior_logits: chex.Array, value: chex.Array, embedding: chex.Array) -> Tree[T]:
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(
      children_prior_logits=batch_update(tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(tree.raw_values, value, node_index),
      node_values=batch_update(tree.node_values, value, node_index),
      node_visits=batch_update(tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(lambda t, s: batch_update(t, s, node_index), tree.embeddings, embedding))
  return tree.replace(**updates)


def update_tree_node_custom(tree: Tree[T], node_index: chex.Array, prior_logits: chex.Array, value: chex.Array, embedding: chex.Array, should_update: chex.Array) -> Tree[T]:
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  new_visit = tree.node_visits[batch_range, node_index] + 1
  
  updated_prior = jnp.where(should_update[:, None], prior_logits, tree.children_prior_logits[batch_range, node_index])
  updated_raw_val = jnp.where(should_update, value, tree.raw_values[batch_range, node_index])
  updated_node_val = jnp.where(should_update, value, tree.node_values[batch_range, node_index])
  updated_visit = jnp.where(should_update, new_visit, tree.node_visits[batch_range, node_index])
  
  updated_embedding = jax.tree.map(
      lambda t, s: jnp.where(
          jnp.reshape(should_update, [-1] + [1] * (len(t.shape) - 2)),
          s,
          t[batch_range, node_index]
      ),
      tree.embeddings,
      embedding
  )
  
  updates = dict(
      children_prior_logits=batch_update(tree.children_prior_logits, updated_prior, node_index),
      raw_values=batch_update(tree.raw_values, updated_raw_val, node_index),
      node_values=batch_update(tree.node_values, updated_node_val, node_index),
      node_visits=batch_update(tree.node_visits, updated_visit, node_index),
      embeddings=jax.tree.map(lambda t, s: batch_update(t, s, node_index), tree.embeddings, updated_embedding)
  )
  return tree.replace(**updates)


def instantiate_tree_from_root(root: base.RootFnOutput, num_simulations: int, root_invalid_actions: chex.Array, extra_data: Any) -> Tree:
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Initialize extra_data as (user_extra_data, depths)
  tree_depths = jnp.zeros(batch_node, dtype=jnp.int32)
  packed_extra_data = (extra_data, tree_depths)

  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=packed_extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree


class _SimulationState(NamedTuple):
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


# Unbatched single simulation to trace the trajectory
def _simulate_single(rng_key, tree, action_selection_fn, max_depth):
  def scan_fn(carry, step_idx):
    node_index, is_continuing, key = carry
    
    key, selection_key = jax.random.split(key)
    action = jax.lax.cond(
        is_continuing,
        lambda: action_selection_fn(selection_key, tree, node_index, step_idx),
        lambda: jnp.zeros((), dtype=jnp.int32)
    )
    next_node_index = tree.children_index[node_index, action]
    
    step_info = (node_index, action, is_continuing)
    
    next_is_continuing = is_continuing & (next_node_index != Tree.UNVISITED)
    next_node_index = jnp.where(is_continuing, next_node_index, node_index)
    
    return (next_node_index, next_is_continuing, key), step_info

  initial_carry = (jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32), True, rng_key)
  (final_node, _, _), trajectory = jax.lax.scan(
      scan_fn,
      initial_carry,
      jnp.arange(max_depth)
  )
  
  trajectory_parents, trajectory_actions, trajectory_active = trajectory
  
  last_active_step_idx = jnp.sum(trajectory_active.astype(jnp.int32)) - 1
  last_active_step_idx = jnp.maximum(last_active_step_idx, 0)
  
  expansion_parent = trajectory_parents[last_active_step_idx]
  expansion_action = trajectory_actions[last_active_step_idx]
  
  return expansion_parent, expansion_action, trajectory_parents, trajectory_actions, trajectory_active


# Vmapped simulation over batch
simulate = jax.vmap(_simulate_single, in_axes=(0, 0, None, None), out_axes=0)


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    sim_node_index: chex.Array,
    state_equal_fn: Callable,
    merge_mode: str = "depth_dependent"
) -> Tuple[Tree[T], chex.Array]:
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  
  next_node_index = tree.children_index[batch_range, parent_index, action]
  is_unvisited = next_node_index == Tree.UNVISITED
  temp_next_node_index = jnp.where(is_unvisited, sim_node_index, next_node_index)

  # Fetch parent depth
  tree_depths = tree.extra_data[1]
  parent_depths = tree_depths[batch_range, parent_index]
  target_depths = parent_depths + 1
  
  embedding = jax.tree.map(lambda x: x[batch_range, parent_index], tree.embeddings)
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  
  parent_is_chance = jax.vmap(lambda t, idx: ~t.is_decision_node[idx])(tree.embeddings, parent_index)
  existing_env_states = tree.embeddings.state_embedding[0]
  new_env_state = embedding.state_embedding[0]
  
  def find_matching_node(existing_states, new_state, valid_mask, state_equal_fn):
    equals = jax.vmap(state_equal_fn, in_axes=(0, None))(existing_states, new_state)
    matches = equals & valid_mask
    found = jnp.any(matches)
    matched_idx = jnp.argmax(matches)
    return found, matched_idx

  if merge_mode == "depth_dependent":
    valid_mask = (tree.node_visits > 0) & tree.embeddings.is_decision_node & (tree_depths == target_depths[:, None])
  elif merge_mode == "depth_independent":
    valid_mask = (tree.node_visits > 0) & tree.embeddings.is_decision_node
  else:
    # Disable state merging entirely (e.g. "pure_tree")
    valid_mask = jnp.zeros_like(tree.node_visits, dtype=jnp.bool_)

  vmapped_find = jax.vmap(
      functools.partial(find_matching_node, state_equal_fn=state_equal_fn),
      in_axes=(0, 0, 0)
  )
  found, matched_idx = vmapped_find(existing_env_states, new_env_state, valid_mask)
  
  should_merge = parent_is_chance & found & is_unvisited
  actual_next_node_index = jnp.where(should_merge, matched_idx, temp_next_node_index)
  
  # Update depths
  new_depths = jnp.where(should_merge, tree_depths[batch_range, matched_idx], target_depths)
  updated_tree_depths = jnp.where(is_unvisited, batch_update(tree_depths, new_depths, actual_next_node_index), tree_depths)
  updated_extra_data = (tree.extra_data[0], updated_tree_depths)
  tree = tree.replace(extra_data=updated_extra_data)
  
  should_update = is_unvisited & ~should_merge
  tree = update_tree_node_custom(tree, actual_next_node_index, step.prior_logits, step.value, embedding, should_update)
  
  tree = tree.replace(
      children_index=jnp.where(is_unvisited, batch_update(tree.children_index, actual_next_node_index, parent_index, action), tree.children_index),
      children_rewards=jnp.where(is_unvisited, batch_update(tree.children_rewards, step.reward, parent_index, action), tree.children_rewards),
      children_discounts=jnp.where(is_unvisited, batch_update(tree.children_discounts, step.discount, parent_index, action), tree.children_discounts),
      parents=jnp.where(is_unvisited, batch_update(tree.parents, parent_index, actual_next_node_index), tree.parents),
      action_from_parent=jnp.where(is_unvisited, batch_update(tree.action_from_parent, action, actual_next_node_index), tree.action_from_parent))
  return tree, actual_next_node_index


# Unbatched backpropagation along trajectory
def _backward_single(tree, parents, actions, active, leaf_node_idx, p=1.0):
  leaf_value = tree.node_values[leaf_node_idx]
  
  rev_parents = jnp.flip(parents)
  rev_actions = jnp.flip(actions)
  rev_active = jnp.flip(active)
  
  def scan_fn(carry, step_data):
    tree, leaf_value = carry
    parent, action, step_active = step_data
    
    new_children_visits = tree.children_visits[parent].at[action].add(1)
    new_children_values = tree.children_values[parent].at[action].set(leaf_value)
    new_parent_visits = tree.node_visits[parent] + 1
    
    Q_values = tree.children_rewards[parent] + tree.children_discounts[parent] * new_children_values
    
    if p == "inf":
      parent_value = jnp.max(Q_values)
    else:
      p_val = float(p) if isinstance(p, str) else p
      sum_val = jnp.sum((new_children_visits / (new_parent_visits + 1e-8)) * jnp.power(jnp.maximum(Q_values, 0.0), p_val))
      parent_value = jnp.power(sum_val, 1.0 / p_val)
    
    updated_node_values = jnp.where(step_active, parent_value, tree.node_values[parent])
    updated_node_visits = jnp.where(step_active, new_parent_visits, tree.node_visits[parent])
    updated_children_values = jnp.where(step_active, new_children_values, tree.children_values[parent])
    updated_children_visits = jnp.where(step_active, new_children_visits, tree.children_visits[parent])
    
    tree = tree.replace(
        node_values=tree.node_values.at[parent].set(updated_node_values),
        node_visits=tree.node_visits.at[parent].set(updated_node_visits),
        children_values=tree.children_values.at[parent].set(updated_children_values),
        children_visits=tree.children_visits.at[parent].set(updated_children_visits)
    )
    
    next_leaf_value = jnp.where(step_active, parent_value, leaf_value)
    return (tree, next_leaf_value), None

  initial_carry = (tree, leaf_value)
  (final_tree, _), _ = jax.lax.scan(
      scan_fn,
      initial_carry,
      (rev_parents, rev_actions, rev_active)
  )
  return final_tree


# Vmapped backpropagation
backward_trajectory = jax.vmap(_backward_single, in_axes=(0, 0, 0, 0, 0, None), out_axes=0)


def search_custom(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    state_equal_fn: Callable,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    merge_mode: str = "depth_dependent",
    p: float = 1.0) -> Tree:
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )
  batch_size = root.value.shape[0]
  if max_depth is None:
    max_depth = 200
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    simulate_keys = jax.random.split(simulate_key, batch_size)
    
    # 1. Simulate and record trajectory
    parent_index, action, trajectory_parents, trajectory_actions, trajectory_active = simulate(
        simulate_keys, tree, action_selection_fn, 2 * max_depth
    )
    
    # 2. Expand and resolve node indices (with state merging)
    sim_node_index = jnp.full((batch_size,), sim + 1, dtype=jnp.int32)
    tree, actual_next_node_index = expand(
        params, expand_key, tree, recurrent_fn, parent_index, action, sim_node_index, state_equal_fn, merge_mode=merge_mode
    )
    
    # 3. Backpropagate along trajectory (custom SP-UCT backpropagation)
    tree = backward_trajectory(
        tree, trajectory_parents, trajectory_actions, trajectory_active, actual_next_node_index, p
    )
    
    loop_state = rng_key, tree
    return loop_state

  tree = instantiate_tree_from_root(root, num_simulations, root_invalid_actions=invalid_actions, extra_data=extra_data)
  _, tree = loop_fn(0, num_simulations, body_fun, (rng_key, tree))
  return tree


def sp_uct_action_selection(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    ucb_mode: str = "spuct",
) -> chex.Array:
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  
  if ucb_mode == "spuct":
    policy_score = pb_c_init * jnp.power(node_visit + 1.0, 0.25) / (jnp.sqrt(visit_counts) + 1e-5)
  else:
    policy_score = pb_c_init * jnp.sqrt(node_visit + 1.0) / (visit_counts + 1e-5)
  
  value_score = qtransform(tree, node_index)
  node_noise_score = 1e-7 * jax.random.uniform(rng_key, (tree.num_actions,))
  to_argmax = value_score + policy_score + node_noise_score
  return action_selection.masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))


def custom_stochastic_muzero_policy(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    decision_recurrent_fn: base.DecisionRecurrentFn,
    chance_recurrent_fn: base.ChanceRecurrentFn,
    num_simulations: int,
    state_equal_fn: Callable,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.6,
    dirichlet_alpha: chex.Numeric = 1.0,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0,
    ucb_mode: str = "spuct",
    merge_mode: str = "depth_dependent",
    p: float = 1.0) -> base.PolicyOutput[None]:
  num_actions = root.prior_logits.shape[-1]
  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
  root = root.replace(prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  batch_size = jax.tree_util.tree_leaves(root.embedding)[0].shape[0]
  dummy_action = jnp.zeros([batch_size], dtype=jnp.int32)
  dummy_output, dummy_afterstate_embedding = decision_recurrent_fn(
      params, rng_key, dummy_action, root.embedding)
  num_chance_outcomes = dummy_output.chance_logits.shape[-1]

  root = root.replace(
      prior_logits=jnp.concatenate([
          root.prior_logits,
          jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
      ], axis=-1),
      embedding=base.StochasticRecurrentState(
          state_embedding=root.embedding,
          afterstate_embedding=dummy_afterstate_embedding,
          is_decision_node=jnp.ones([batch_size], dtype=bool)))

  recurrent_fn = _make_stochastic_recurrent_fn(
      decision_node_fn=decision_recurrent_fn,
      chance_node_fn=chance_recurrent_fn,
      num_actions=num_actions,
      num_chance_outcomes=num_chance_outcomes,
  )

  interior_decision_node_selection_fn = functools.partial(
      sp_uct_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform,
      ucb_mode=ucb_mode)

  interior_action_selection_fn = _make_stochastic_action_selection_fn(
      interior_decision_node_selection_fn, num_actions)

  root_action_selection_fn = functools.partial(interior_action_selection_fn, depth=0)

  search_tree = search_custom(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      state_equal_fn=state_equal_fn,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn,
      merge_mode=merge_mode,
      p=p)

  search_tree = _mask_tree(search_tree, num_actions, 'decision')
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = _apply_temperature(_get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(action=action, action_weights=action_weights, search_tree=search_tree)


# =====================================================================
# Unified Monolithic Search Entrypoint
# =====================================================================

def jax_mctx_search(
    key: jax.Array,
    obs: jax.Array,
    state: Any,
    env_step: Callable,
    env_reset: Callable,
    action_mask_fn: Optional[Callable],
    reward_norm_fn: Optional[Callable],
    state_equal_fn: Optional[Callable],
    num_actions: int,
    num_simulations: int = 100,
    max_depth: Optional[int] = None,
    gamma: float = 0.99,
    rollout_depth: int = 10,
    num_chance_outcomes: int = 4,
    nn_model: Optional[Any] = None,
    nn_params: Optional[Any] = None,
    ucb_mode: str = "spuct",
    merge_mode: str = "depth_dependent",
    p: Any = 1.0,
    c: Any = 1.25,
    return_weights: bool = False,
    return_node_count: bool = False,
):
    """
    Unified MCTS Engine.
    Operates on a single environment state (Can be vmap'ed externally).
    """
    # Since solver_fn is vmapped, expand to batch size 1
    batched_obs = jax.tree.map(lambda x: jnp.expand_dims(x, 0), obs)
    batched_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
    
    key_root, key_search = jax.random.split(key)

    # 1. Root Initialization
    if nn_model is not None and nn_params is not None:
        root_logits, root_value = nn_model.apply(nn_params, batched_obs)
        if action_mask_fn is not None:
            mask = action_mask_fn(obs)
            root_logits = jnp.where(mask[None, :], root_logits, -1e9)
    else:
        if action_mask_fn is not None:
            mask = action_mask_fn(obs)
            root_logits = jnp.where(mask, 0.0, -1e9)[None, :]
        else:
            root_logits = jnp.zeros((num_actions,))[None, :]
        
        # Rollout for root node
        root_value = run_random_rollout(
            key_root, state, obs, env_step, reward_norm_fn, num_actions, rollout_depth, action_mask_fn
        )
        root_value = jnp.array([root_value])

    # Wrap root specifications for mctx
    root = mctx.RootFnOutput(
        prior_logits=root_logits,
        value=root_value,
        embedding=(batched_state, batched_obs)
    )

    # 2. Define decision and chance recurrent functions
    def decision_recurrent_fn(params, rng_key, action, state_embedding):
        state, obs = state_embedding
        batch_size = action.shape[0]
        
        chance_logits = jnp.zeros((batch_size, num_chance_outcomes))
        afterstate_value = jnp.zeros((batch_size,))
        afterstate_embedding = (state, obs, action)
        
        return mctx.DecisionRecurrentFnOutput(
            chance_logits=chance_logits,
            afterstate_value=afterstate_value
        ), afterstate_embedding

    def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate_embedding):
        state, obs, action = afterstate_embedding
        
        def single_step(key, s, o, a, co):
            step_key, rollout_key = jax.random.split(key)
            
            # Artificial branching: fold chance_outcome into environment state's PRNGKey
            if hasattr(s, "key"):
                s_key = jax.random.fold_in(s.key, co)
                s = s.replace(key=s_key)
            else:
                # If environment state has no key (e.g. gymnax), fold into step key
                step_key = jax.random.fold_in(step_key, co)
            
            next_s, next_o, reward, done, _ = env_step(step_key, s, a)
            
            # Value/Prior evaluation at decision node
            if nn_model is not None and params is not None:
                policy_logits, state_value = nn_model.apply(params, next_o)
                if action_mask_fn is not None:
                    mask = action_mask_fn(next_o)
                    policy_logits = jnp.where(mask, policy_logits, -1e9)
            else:
                if action_mask_fn is not None:
                    mask = action_mask_fn(next_o)
                    prior_logits = jnp.where(mask, 0.0, -1e9)
                else:
                    prior_logits = jnp.zeros((num_actions,))
                    
                state_value = run_random_rollout(
                    rollout_key, next_s, next_o, env_step, reward_norm_fn, num_actions, rollout_depth, action_mask_fn
                )
                policy_logits = prior_logits

            discount = jnp.where(done, 0.0, gamma)
            norm_reward = reward_norm_fn(reward) if reward_norm_fn is not None else reward
            state_value = jnp.where(done, 0.0, state_value)
            
            return (next_s, next_o), policy_logits, state_value, norm_reward, discount
 
        batch_size = chance_outcome.shape[0]
        keys = jax.random.split(rng_key, batch_size)
        vmapped_fn = jax.vmap(single_step, in_axes=(0, 0, 0, 0, 0))
        next_state_embedding, action_logits, values, rewards, discounts = vmapped_fn(
            keys, state, obs, action, chance_outcome
        )
        
        return mctx.ChanceRecurrentFnOutput(
            action_logits=action_logits,
            value=values,
            reward=rewards,
            discount=discounts
        ), next_state_embedding

    if max_depth is None:
        import math
        if gamma >= 1.0:
            max_depth = 10
        else:
            max_depth = int(math.ceil(math.log(num_simulations) / (2.0 * math.log(1.0 / gamma))))

    # 3. Execute Custom Stochastic Policy Search
    policy_output = custom_stochastic_muzero_policy(
        params=nn_params,
        rng_key=key_search,
        root=root,
        decision_recurrent_fn=decision_recurrent_fn,
        chance_recurrent_fn=chance_recurrent_fn,
        num_simulations=num_simulations,
        state_equal_fn=state_equal_fn,
        max_depth=max_depth,
        qtransform=mctx.qtransform_by_parent_and_siblings,
        ucb_mode=ucb_mode,
        merge_mode=merge_mode,
        p=p,
        pb_c_init=c
    )

    if return_weights:
        return policy_output.action[0], policy_output.action_weights[0]
    if return_node_count:
        visited = jnp.sum(policy_output.search_tree.node_visits[0] > 0)
        return policy_output.action[0], visited
    return policy_output.action[0]