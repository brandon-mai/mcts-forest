import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Callable

# Define a static maximum size for our allocation based on the budget
NUM_SIMULATIONS = 100
MAX_NODES = NUM_SIMULATIONS + 2

class MCTSTree(NamedTuple):
    node_states: Any          # PyTree matching the environment's state structure
    node_obs: Any            # PyTree matching the environment's observation structure
    node_visits: jnp.ndarray  # Shape: (MAX_NODES,)
    node_values: jnp.ndarray  # Shape: (MAX_NODES,)
    child_index: jnp.ndarray  # Shape: (MAX_NODES, num_actions)
    num_nodes: jnp.ndarray    # Scalar tracker for allocated nodes


def mcts_search(
    key: jax.Array,
    obs: Any,
    state: Any,
    env_step: Callable,
    env_reset: Callable,
    action_mask_fn: Callable,
    reward_norm_fn: Callable,
    state_equal_fn: Callable,
    num_actions: int,
    c_exploration: float = 1.414
) -> jnp.ndarray:
    """
    Performs a pure MCTS search from a given state using random rollouts 
    and a transposition table lookup for graph-structured states.
    """
    
    # --- 1. Initialization ---
    # Pre-allocate flat arrays representing our tree structure
    init_node_states = jax.tree.map(lambda x: jnp.tile(x, (MAX_NODES, *([1] * x.ndim))), state)
    init_node_obs = jax.tree.map(lambda x: jnp.tile(x, (MAX_NODES, *([1] * x.ndim))), obs)
    
    tree = MCTSTree(
        node_states=init_node_states,
        node_obs=init_node_obs,
        node_visits=jnp.zeros((MAX_NODES,), dtype=jnp.int32),
        node_values=jnp.zeros((MAX_NODES,), dtype=jnp.float32),
        child_index=jnp.full((MAX_NODES, num_actions), -1, dtype=jnp.int32),
        num_nodes=jnp.array(1, dtype=jnp.int32)  # Root is at index 0
    )
    
    # Initialize the root node statistics
    tree = tree._replace(
        node_visits=tree.node_visits.at[0].set(1)
    )

    # --- 2. Simulation Loop Helper Functions ---
    
    def rollout(rollout_key: jax.Array, start_state: Any, start_obs: Any) -> jnp.float32:
        """Executes a random rollout until a terminal state is reached."""
        def cond_fn(loop_state):
            _, _, _, done, _ = loop_state
            return ~done

        def body_fn(loop_state):
            curr_key, s, o, _, total_reward, discount = loop_state
            step_key, next_key = jax.random.split(curr_key)
            
            mask = action_mask_fn(o)
            logits = jnp.where(mask, 0.0, -jnp.inf)
            action = jax.random.categorical(step_key, logits)
            
            next_s, next_o, reward, done = env_step(s, action)
            return next_key, next_s, next_o, done, total_reward + discount * reward, discount * 1.0

        _, _, _, _, final_return, _ = jax.lax.while_loop(
            cond_fn, body_fn, (rollout_key, start_state, start_obs, False, 0.0, 1.0)
        )
        return reward_norm_fn(final_return)

    def run_one_simulation(sim_idx: int, loop_state) -> tuple:
        curr_tree, sim_key = loop_state
        select_key, rollout_key = jax.random.split(sim_key)
        
        # --- PHASE 1: Selection ---
        def select_cond(state_tuple):
            idx, _, hit_unexpanded = state_tuple
            return (idx >= 0) & (~hit_unexpanded)

        def select_body(state_tuple):
            idx, path, _ = state_tuple
            mask = action_mask_fn(jax.tree.map(lambda x: x[idx], curr_tree.node_obs))
            
            # Classical UCT formula implementation
            parent_n = curr_tree.node_visits[idx]
            child_indices = curr_tree.child_index[idx]
            child_n = curr_tree.node_visits[child_indices]
            child_q = curr_tree.node_values[child_indices] / (child_n + 1e-6)
            
            exploration = c_exploration * jnp.sqrt(jnp.log(parent_n + 1) / (child_n + 1e-6))
            uct_score = jnp.where(child_n == 0, jnp.inf, child_q + exploration)
            uct_score = jnp.where(mask, uct_score, -jnp.inf)
            
            chosen_action = jnp.argmax(uct_score)
            next_idx = child_indices[chosen_action]
            
            # Record choice along path array
            next_path = path.at[idx].set(chosen_action)
            return next_idx, next_path, next_idx == -1

        # Track choices made down the selection trajectory
        initial_path = jfull_path = jnp.full((MAX_NODES,), -1, dtype=jnp.int32)
        leaf_idx, selection_path, _ = jax.lax.while_loop(
            select_cond, select_body, (0, initial_path, False)
        )
        
        # Determine the parent node where selection broke out
        parent_node_idx = jnp.argmax(selection_path >= 0) 
        chosen_act = selection_path[parent_node_idx]
        parent_state = jax.tree.map(lambda x: x[parent_node_idx], curr_tree.node_states)
        
        # --- PHASE 2: Expansion & Transposition Lookup ---
        next_s, next_o, immediate_reward, done = env_step(parent_state, chosen_act)
        
        # Vectorized Transposition Check via state_equal_fn
        is_match = jax.vmap(lambda s: state_equal_fn(s, next_s))(curr_tree.node_states)
        is_valid_match = is_match & (jnp.arange(MAX_NODES) < curr_tree.num_nodes)
        has_transposition = jnp.any(is_valid_match)
        matched_node_idx = jnp.argmax(is_valid_match)
        
        # Determine target allocation node index
        allocated_idx = jnp.where(has_transposition, matched_node_idx, curr_tree.num_nodes)
        
        # Update Tree node payload arrays if unique
        new_num_nodes = curr_tree.num_nodes + jnp.where(has_transposition, 0, 1)
        updated_states = jax.tree.map(lambda t, s: jnp.where(has_transposition, t, t.at[allocated_idx].set(s)), curr_tree.node_states, next_s)
        updated_obs = jax.tree.map(lambda t, o: jnp.where(has_transposition, t, t.at[allocated_idx].set(o)), curr_tree.node_obs, next_o)
        
        # Link parent to child node pointer index
        updated_child_index = curr_tree.child_index.at[parent_node_idx, chosen_act].set(allocated_idx)
        
        # --- PHASE 3: Simulation (Rollout) ---
        rollout_value = jnp.where(done, reward_norm_fn(immediate_reward), rollout(rollout_key, next_s, next_o))
        
        # --- PHASE 4: Backpropagation ---
        # Traverse up selection path updating visited nodes
        def backprop_node(idx, tree_accum):
            is_visited = selection_path[idx] >= 0
            t_visits = tree_accum.node_visits.at[idx].add(jnp.where(is_visited, 1, 0))
            t_values = tree_accum.node_values.at[idx].add(jnp.where(is_visited, rollout_value, 0.0))
            return tree_accum._replace(node_visits=t_visits, node_values=t_values)
            
        curr_tree = curr_tree._replace(
            node_states=updated_states,
            node_obs=updated_obs,
            child_index=updated_child_index,
            num_nodes=new_num_nodes
        )
        
        curr_tree = jax.lax.fori_loop(0, MAX_NODES, backprop_node, curr_tree)
        
        return curr_tree, select_key

    # --- 3. Run Search ---
    final_tree, _ = jax.lax.fori_loop(
        0, NUM_SIMULATIONS, run_one_simulation, (tree, key)
    )
    
    # --- 4. Choose Action ---
    # Return the action index with the maximum visit counts at the root level
    root_child_indices = final_tree.child_index[0]
    root_child_visits = final_tree.node_visits[root_child_indices]
    # Filter unexpanded paths (-1 indexes array wrapping edge bounds safely)
    root_child_visits = jnp.where(root_child_indices == -1, -1, root_child_visits)
    
    return jnp.argmax(root_child_visits)