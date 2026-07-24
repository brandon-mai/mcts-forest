import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Callable, Any

# =====================================================================
# 1. Frozen Lake (8x8 map from JAIR_cpp_v5)
# =====================================================================

class FrozenLakeState(NamedTuple):
    pos: jnp.ndarray  # int32
    time: jnp.ndarray # int32

HOLE_MASK = jnp.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
], dtype=jnp.bool_)

def make_frozen_lake_fns():
    def reset_fn(key: jax.Array) -> Tuple[FrozenLakeState, jnp.ndarray]:
        state = FrozenLakeState(pos=jnp.int32(0), time=jnp.int32(0))
        return state, state.pos

    def step_fn(key: jax.Array, state: FrozenLakeState, action: jnp.ndarray) -> Tuple[FrozenLakeState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        # Slippery transition
        slip_key, _ = jax.random.split(key)
        slip = jax.random.randint(slip_key, (), 0, 3) # 0, 1, or 2
        eff_act = (action + slip - 1) % 4
        
        row = state.pos // 8
        col = state.pos % 8
        
        # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
        new_col = jnp.where(eff_act == 0, jnp.maximum(0, col - 1), col)
        new_row = jnp.where(eff_act == 1, jnp.minimum(7, row + 1), row)
        new_col = jnp.where(eff_act == 2, jnp.minimum(7, col + 1), new_col)
        new_row = jnp.where(eff_act == 3, jnp.maximum(0, row - 1), new_row)
        
        new_pos = new_row * 8 + new_col
        next_time = state.time + 1
        
        # Check if previous state was already terminal
        prev_is_goal = (state.pos == 63)
        prev_is_hole = HOLE_MASK[state.pos]
        was_terminal = prev_is_goal | prev_is_hole | (state.time >= 200)
        
        pos_to_use = jnp.where(was_terminal, state.pos, new_pos)
        time_to_use = jnp.where(was_terminal, state.time, next_time)
        
        is_goal = (pos_to_use == 63)
        is_hole = HOLE_MASK[pos_to_use]
        
        reward = jnp.where(is_goal & ~was_terminal, 1.0, 0.0)
        done = is_goal | is_hole | (time_to_use >= 200)
        
        next_state = FrozenLakeState(pos=pos_to_use, time=time_to_use)
        return next_state, next_state.pos, reward, done, {}

    action_mask_fn = lambda obs: jnp.ones(4, dtype=jnp.bool_)
    reward_norm_fn = lambda r: r
    state_equal_fn = lambda s1, s2: (s1.pos == s2.pos)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, 4


# =====================================================================
# 2. Passenger Grid (7x6 grid, 3 passengers, time limit 50)
# =====================================================================

class PassengerGridState(NamedTuple):
    x: jnp.ndarray              # int32
    y: jnp.ndarray              # int32
    passenger_mask: jnp.ndarray # int32
    time: jnp.ndarray           # int32

def make_passenger_grid_fns():
    def reset_fn(key: jax.Array) -> Tuple[PassengerGridState, jnp.ndarray]:
        state = PassengerGridState(x=jnp.int32(0), y=jnp.int32(0), passenger_mask=jnp.int32(0), time=jnp.int32(0))
        spatial_id = state.y * 7 + state.x
        obs = (((spatial_id << 3) | state.passenger_mask) * 51) + state.time
        return state, obs

    def step_fn(key: jax.Array, state: PassengerGridState, action: jnp.ndarray) -> Tuple[PassengerGridState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        # Slippery probabilities: [0.25 (left), 0.5 (intended), 0.25 (right)]
        slip_key, _ = jax.random.split(key)
        u = jax.random.uniform(slip_key)
        slip_offset = jnp.where(u < 0.25, -1, jnp.where(u < 0.75, 0, 1))
        eff_act = (action + slip_offset + 4) % 4

        # Movement: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
        nx = jnp.where(eff_act == 0, jnp.maximum(0, state.x - 1), state.x)
        ny = jnp.where(eff_act == 1, jnp.minimum(5, state.y + 1), state.y)
        nx = jnp.where(eff_act == 2, jnp.minimum(6, state.x + 1), nx)
        ny = jnp.where(eff_act == 3, jnp.maximum(0, state.y - 1), ny)

        # Passenger pickup: P0(1,2), P1(0,5), P2(6,4)
        p0_match = (nx == 1) & (ny == 2)
        p1_match = (nx == 0) & (ny == 5)
        p2_match = (nx == 6) & (ny == 4)
        
        new_mask = state.passenger_mask | jnp.where(p0_match, 1, 0) | jnp.where(p1_match, 2, 0) | jnp.where(p2_match, 4, 0)
        next_time = state.time + 1

        is_goal = (nx == 6) & (ny == 0)
        
        # Count picked passengers (popcount)
        picked = ((new_mask & 1) > 0).astype(jnp.int32) + ((new_mask & 2) > 0).astype(jnp.int32) + ((new_mask & 4) > 0).astype(jnp.int32)
        goal_rewards = jnp.array([0.0, 1.0, 3.0, 7.0], dtype=jnp.float32)
        goal_reward = goal_rewards[picked]

        reward = jnp.where(is_goal, goal_reward, 0.0)
        done = is_goal | (next_time == 50)

        next_state = PassengerGridState(x=nx, y=ny, passenger_mask=new_mask, time=next_time)
        spatial_id = next_state.y * 7 + next_state.x
        obs = (((spatial_id << 3) | next_state.passenger_mask) * 51) + next_state.time
        return next_state, obs, reward, done, {}

    action_mask_fn = lambda obs: jnp.ones(4, dtype=jnp.bool_)
    reward_norm_fn = lambda r: r / 7.0
    state_equal_fn = lambda s1, s2: (s1.x == s2.x) & (s1.y == s2.y) & (s1.passenger_mask == s2.passenger_mask)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, 4


# =====================================================================
# 3. Factored River Swim (4 rivers, 8 locations, time limit 35)
# =====================================================================

class FactoredRiverSwimState(NamedTuple):
    positions: jnp.ndarray # int32 array shape (4,)
    time: jnp.ndarray      # int32

def make_factored_river_swim_fns():
    def reset_fn(key: jax.Array) -> Tuple[FactoredRiverSwimState, jnp.ndarray]:
        state = FactoredRiverSwimState(positions=jnp.zeros(4, dtype=jnp.int32), time=jnp.int32(0))
        spatial_id = state.positions[0] + state.positions[1]*8 + state.positions[2]*64 + state.positions[3]*512
        obs = spatial_id * 36 + state.time
        return state, obs

    def step_fn(key: jax.Array, state: FactoredRiverSwimState, action: jnp.ndarray) -> Tuple[FactoredRiverSwimState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        action_bits = jnp.array([(action >> i) & 1 for i in range(4)], dtype=jnp.int32)
        
        # Split key for 4 rivers
        subkeys = jax.random.split(key, 4)
        u_samples = jax.vmap(lambda k: jax.random.uniform(k))(subkeys)

        def step_river(pos, act, u):
            # act == 0 (REST): max(0, pos - 1)
            rest_pos = jnp.maximum(0, pos - 1)
            
            # act == 1 (SWIM):
            swim_at_0 = jnp.where(u < 0.4, 0, 1)
            swim_at_7 = jnp.where(u < 0.4, 6, 7)
            swim_mid = jnp.where(u < 0.05, pos - 1, jnp.where(u < 0.65, pos, pos + 1))
            
            swim_pos = jnp.where(pos == 0, swim_at_0, jnp.where(pos == 7, swim_at_7, swim_mid))
            return jnp.where(act == 0, rest_pos, swim_pos)

        next_positions = jax.vmap(step_river)(state.positions, action_bits, u_samples)
        
        # Rewards:
        # pos == 0 and act == 0 -> +0.1
        # pos == 7 and act == 1 -> +1.0
        # all pos == 7 and all act == 1 -> +4.0
        rest_r = jnp.where((state.positions == 0) & (action_bits == 0), 0.1, 0.0)
        swim_r = jnp.where((state.positions == 7) & (action_bits == 1), 1.0, 0.0)
        all_goal_swim = jnp.all(state.positions == 7) & jnp.all(action_bits == 1)
        bonus = jnp.where(all_goal_swim, 4.0, 0.0)
        
        reward = (jnp.sum(rest_r + swim_r) + bonus) / 8.0
        next_time = state.time + 1
        done = (next_time == 35)

        next_state = FactoredRiverSwimState(positions=next_positions, time=next_time)
        spatial_id = next_state.positions[0] + next_state.positions[1]*8 + next_state.positions[2]*64 + next_state.positions[3]*512
        obs = spatial_id * 36 + next_state.time
        return next_state, obs, reward, done, {}

    action_mask_fn = lambda obs: jnp.ones(16, dtype=jnp.bool_)
    reward_norm_fn = lambda r: r
    state_equal_fn = lambda s1, s2: jnp.all(s1.positions == s2.positions)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, 16


# =====================================================================
# 4. SysAdmin Ring (20 machines, 21 actions, time limit 50)
# =====================================================================

class SysAdminRingState(NamedTuple):
    alive_mask: jnp.ndarray # int32
    time: jnp.ndarray       # int32

def make_sysadmin_ring_fns():
    def reset_fn(key: jax.Array) -> Tuple[SysAdminRingState, jnp.ndarray]:
        state = SysAdminRingState(alive_mask=jnp.int32(0), time=jnp.int32(0))
        obs = state.alive_mask * 51 + state.time
        return state, obs

    def step_fn(key: jax.Array, state: SysAdminRingState, action: jnp.ndarray) -> Tuple[SysAdminRingState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        keys = jax.random.split(key, 20)
        u_samples = jax.vmap(lambda k: jax.random.uniform(k))(keys)
        
        machines = jnp.arange(20, dtype=jnp.int32)
        prev_machines = (machines + 19) % 20
        
        prev_running = ((state.alive_mask >> prev_machines) & 1) == 1
        self_running = ((state.alive_mask >> machines) & 1) == 1
        
        # Transitions probabilities:
        # ~prev & ~self: 0.0238
        #  prev & ~self: 0.0475
        # ~prev &  self: 0.525
        #  prev &  self: 0.95
        p = jnp.where(~prev_running & ~self_running, 0.0238,
            jnp.where(prev_running & ~self_running, 0.0475,
            jnp.where(~prev_running & self_running, 0.525, 0.95)))
            
        p_eff = jnp.where(action == machines, 1.0, p)
        next_bits = (u_samples < p_eff).astype(jnp.int32)
        
        shifts = jnp.arange(20, dtype=jnp.int32)
        next_alive_mask = jnp.sum(next_bits << shifts)
        
        # Reward: fraction of running computers
        num_running = jnp.sum(next_bits)
        reward = num_running.astype(jnp.float32) / 20.0
        
        next_time = state.time + 1
        done = (next_time == 50)
        
        next_state = SysAdminRingState(alive_mask=next_alive_mask, time=next_time)
        obs = next_state.alive_mask * 51 + next_state.time
        return next_state, obs, reward, done, {}

    action_mask_fn = lambda obs: jnp.ones(21, dtype=jnp.bool_)
    reward_norm_fn = lambda r: r
    state_equal_fn = lambda s1, s2: (s1.alive_mask == s2.alive_mask)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, 21


# =====================================================================
# 5. Four Rooms (Custom 11x11, randomized doors/start/goal from JAIR_cpp_v5)
# =====================================================================

class FourRoomsState(NamedTuple):
    x: jnp.ndarray        # int32
    y: jnp.ndarray        # int32
    start_x: jnp.ndarray  # int32
    start_y: jnp.ndarray  # int32
    goal_x: jnp.ndarray   # int32
    goal_y: jnp.ndarray   # int32
    door0_y: jnp.ndarray  # int32
    door1_y: jnp.ndarray  # int32
    door2_x: jnp.ndarray  # int32
    door3_x: jnp.ndarray  # int32
    time: jnp.ndarray     # int32

def make_four_rooms_custom_fns():
    def reset_fn(key: jax.Array) -> Tuple[FourRoomsState, jnp.ndarray]:
        k_d0, k_d1, k_d2, k_d3, k_start, k_goal = jax.random.split(key, 6)
        
        door0_y = jax.random.randint(k_d0, (), 0, 5)        # wall x=5, y in [0, 4]
        door1_y = 6 + jax.random.randint(k_d1, (), 0, 5)    # wall x=5, y in [6, 10]
        door2_x = jax.random.randint(k_d2, (), 0, 5)        # wall y=5, x in [0, 4]
        door3_x = 6 + jax.random.randint(k_d3, (), 0, 5)    # wall y=5, x in [6, 10]
        
        # 121 grid cells
        all_x = jnp.tile(jnp.arange(11, dtype=jnp.int32), 11)
        all_y = jnp.repeat(jnp.arange(11, dtype=jnp.int32), 11)
        
        is_wall = ((all_x == 5) | (all_y == 5)) & ~((all_x == 5) & (all_y == door0_y)) & ~((all_x == 5) & (all_y == door1_y)) & ~((all_x == door2_x) & (all_y == 5)) & ~((all_x == door3_x) & (all_y == 5))
        valid_mask = ~is_wall
        
        probs = valid_mask.astype(jnp.float32) / jnp.sum(valid_mask.astype(jnp.float32))
        start_idx = jax.random.choice(k_start, 121, p=probs)
        
        # Goal must be different from start
        goal_probs = probs.at[start_idx].set(0.0)
        goal_probs = goal_probs / jnp.sum(goal_probs)
        goal_idx = jax.random.choice(k_goal, 121, p=goal_probs)
        
        start_x = all_x[start_idx]
        start_y = all_y[start_idx]
        goal_x = all_x[goal_idx]
        goal_y = all_y[goal_idx]
        
        state = FourRoomsState(
            x=start_x, y=start_y, start_x=start_x, start_y=start_y,
            goal_x=goal_x, goal_y=goal_y,
            door0_y=door0_y, door1_y=door1_y, door2_x=door2_x, door3_x=door3_x,
            time=jnp.int32(0)
        )
        cell_id = state.y * 11 + state.x
        obs = cell_id * 51 + state.time
        return state, obs

    def step_fn(key: jax.Array, state: FourRoomsState, action: jnp.ndarray) -> Tuple[FourRoomsState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        # Slippery probabilities: [0.25 (left), 0.5 (intended), 0.25 (right)]
        slip_key, _ = jax.random.split(key)
        u = jax.random.uniform(slip_key)
        slip_offset = jnp.where(u < 0.25, -1, jnp.where(u < 0.75, 0, 1))
        eff_act = (action + slip_offset + 4) % 4
        
        dx = jnp.where(eff_act == 0, -1, jnp.where(eff_act == 2, 1, 0))
        dy = jnp.where(eff_act == 3, -1, jnp.where(eff_act == 1, 1, 0))
        
        nx = state.x + dx
        ny = state.y + dy
        
        in_bounds = (nx >= 0) & (nx <= 10) & (ny >= 0) & (ny <= 10)
        is_door = ((nx == 5) & (ny == state.door0_y)) | ((nx == 5) & (ny == state.door1_y)) | ((nx == state.door2_x) & (ny == 5)) | ((nx == state.door3_x) & (ny == 5))
        is_wall = ((nx == 5) | (ny == 5)) & ~is_door
        
        valid_step = in_bounds & ~is_wall
        
        final_x = jnp.where(valid_step, nx, state.x)
        final_y = jnp.where(valid_step, ny, state.y)
        
        next_time = state.time + 1
        is_goal = (final_x == state.goal_x) & (final_y == state.goal_y)
        
        goal_reward = 1.0 - 0.9 * (next_time.astype(jnp.float32) / 50.0)
        reward = jnp.where(is_goal, goal_reward, 0.0)
        done = is_goal | (next_time == 50)
        
        next_state = FourRoomsState(
            x=final_x, y=final_y, start_x=state.start_x, start_y=state.start_y,
            goal_x=state.goal_x, goal_y=state.goal_y,
            door0_y=state.door0_y, door1_y=state.door1_y, door2_x=state.door2_x, door3_x=state.door3_x,
            time=next_time
        )
        cell_id = next_state.y * 11 + next_state.x
        obs = cell_id * 51 + next_state.time
        return next_state, obs, reward, done, {}

    action_mask_fn = lambda obs: jnp.ones(4, dtype=jnp.bool_)
    reward_norm_fn = lambda r: r
    state_equal_fn = lambda s1, s2: (s1.x == s2.x) & (s1.y == s2.y) & (s1.goal_x == s2.goal_x) & (s1.goal_y == s2.goal_y)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, 4


# =====================================================================
# Factory Function
# =====================================================================

def make_custom_fns(env_name: str):
    name = env_name.lower().replace("-", "_").replace(" ", "_")
    if name in ["frozen_lake", "frozenlake"]:
        return make_frozen_lake_fns()
    elif name in ["passenger_grid", "passengergrid"]:
        return make_passenger_grid_fns()
    elif name in ["factored_river_swim", "factoredriverswim", "river_swim", "riverswim"]:
        return make_factored_river_swim_fns()
    elif name in ["sysadmin_ring", "sysadminring", "sysadmin"]:
        return make_sysadmin_ring_fns()
    elif name in ["four_rooms", "fourrooms_custom"]:
        return make_four_rooms_custom_fns()
    else:
        raise ValueError(f"Unknown custom environment name: {env_name}")
