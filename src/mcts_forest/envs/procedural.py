import numpy as np
from numba import njit
import math

# --- Factored River Swim ---

@njit(cache=True)
def river_swim_step(s, a, nr, nl, tl):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    sid = s // (tl + 1)
    
    # Compute reward
    reward = 0.0
    all_at_goal = True
    all_swim_up = True
    goal = nl - 1
    
    # Deconstruct state and compute next sid
    next_sid = 0
    factor = 1
    temp_sid = sid
    for i in range(nr):
        pos = temp_sid % nl
        temp_sid //= nl
        act = (a >> i) & 1
        
        # Individual rewards
        if pos == 0 and act == 0: reward += 0.1
        if pos == goal and act == 1: reward += 1.0
        
        # Bonus conditions
        if pos != goal: all_at_goal = False
        if act != 1: all_swim_up = False
        
        # Transitions
        next_pos = pos
        if act == 0:
            next_pos = max(0, pos - 1)
        else:
            p = np.random.random()
            if pos == 0:
                next_pos = 0 if p < 0.4 else 1
            elif pos == goal:
                next_pos = goal - 1 if p < 0.4 else goal
            else:
                if p < 0.05: next_pos = pos - 1
                elif p < 0.65: next_pos = pos
                else: next_pos = pos + 1
        
        next_sid += next_pos * factor
        factor *= nl
        
    if all_at_goal and all_swim_up:
        reward += float(nr)
    
    reward /= (2.0 * nr)
    next_t = t + 1
    next_s = next_sid * (tl + 1) + next_t
    return next_s, reward, (next_t >= tl)

@njit(cache=True)
def river_swim_rollout(s, nr, nl, tl, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        # Random action
        a = np.random.randint(0, 1 << nr)
        curr_s, r, done = river_swim_step(curr_s, a, nr, nl, tl)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

# --- Four Rooms ---

@njit(cache=True)
def four_rooms_step(s, a, n, gs, tl, slip, doors, gx, gy):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    pos = s // (tl + 1)
    x, y = pos % gs, pos // gs
    
    # Stochasticity
    eff_a = a
    if slip:
        p = np.random.random()
        if p < 0.25: eff_a = (a + 3) % 4 # Left
        elif p >= 0.75: eff_a = (a + 1) % 4 # Right
    
    # Movements: LEFT=0, DOWN=1, RIGHT=2, UP=3
    dx, dy = 0, 0
    if eff_a == 0: dx = -1
    elif eff_a == 1: dy = 1
    elif eff_a == 2: dx = 1
    elif eff_a == 3: dy = -1
    
    nx, ny = x + dx, y + dy
    
    # Wall check
    blocked = False
    if nx < 0 or nx >= gs or ny < 0 or ny >= gs:
        blocked = True
    else:
        # JAIR Wall logic: x=n or y=n is wall, unless it's a door
        is_on_wall_line = (nx == n or ny == n)
        if is_on_wall_line:
            idx = ny * gs + nx
            is_door = False
            for d in doors:
                if idx == d:
                    is_door = True
                    break
            if not is_door:
                blocked = True
    
    if blocked:
        nx, ny = x, y
        
    next_t = t + 1
    reward = 0.0
    done = False
    if nx == gx and ny == gy:
        reward = 1.0 - 0.9 * (float(next_t) / float(tl))
        done = True
    elif next_t >= tl:
        done = True
        
    next_s = (ny * gs + nx) * (tl + 1) + next_t
    return next_s, reward, done

@njit(cache=True)
def four_rooms_rollout(s, n, gs, tl, slip, doors, gx, gy, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 4)
        curr_s, r, done = four_rooms_step(curr_s, a, n, gs, tl, slip, doors, gx, gy)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

# --- Passenger Grid ---

@njit(cache=True)
def passenger_grid_step(s, a, w, h, tl, slip, pass_pos, gx, gy):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    encoded = s // (tl + 1)
    mask = encoded & 7 # 3 passengers
    spatial_id = encoded >> 3
    x, y = spatial_id % w, spatial_id // w
    
    # Stochasticity
    eff_a = a
    if slip:
        p = np.random.random()
        if p < 0.25: eff_a = (a + 3) % 4
        elif p >= 0.75: eff_a = (a + 1) % 4
        
    dx, dy = 0, 0
    if eff_a == 0: dx = -1 # LEFT
    elif eff_a == 1: dy = 1 # DOWN
    elif eff_a == 2: dx = 1 # RIGHT
    elif eff_a == 3: dy = -1 # UP
    
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        nx, ny = x, y
        
    # Pickup
    n_mask = mask
    for i in range(3):
        if nx == pass_pos[i, 0] and ny == pass_pos[i, 1]:
            n_mask |= (1 << i)
            
    next_t = t + 1
    reward = 0.0
    done = False
    if nx == gx and ny == gy:
        # Count bits in n_mask
        picked = 0
        for i in range(3):
            if (n_mask >> i) & 1: picked += 1
        # Use an array for rewards
        rewards_arr = np.array([0.0, 1.0, 3.0, 7.0])
        reward = rewards_arr[picked]
        done = True
    elif next_t >= tl:
        done = True
        
    next_sid = (ny * w + nx) << 3 | n_mask
    next_s = next_sid * (tl + 1) + next_t
    return next_s, reward, done

@njit(cache=True)
def passenger_grid_rollout(s, w, h, tl, slip, pass_pos, gx, gy, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 4)
        curr_s, r, done = passenger_grid_step(curr_s, a, w, h, tl, slip, pass_pos, gx, gy)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

# --- Sysadmin Ring ---

@njit(cache=True)
def sysadmin_ring_step(s, a, nc, tl, probs):
    t = s % (tl + 1)
    if t >= tl:
        return s, 0.0, True
    
    mask = s // (tl + 1)
    
    next_mask = 0
    for i in range(nc):
        if a == i:
            next_mask |= (1 << i)
        else:
            prev_machine = (i - 1 + nc) % nc
            prev_running = (mask >> prev_machine) & 1
            self_running = (mask >> i) & 1
            # probs: [c_c, c_r, r_c, r_r]
            p = probs[(self_running << 1) | prev_running]
            if np.random.random() < p:
                next_mask |= (1 << i)
                
    # Reward: count bits
    count = 0
    for i in range(nc):
        if (next_mask >> i) & 1: count += 1
    reward = float(count) / float(nc)
    
    next_t = t + 1
    next_s = next_mask * (tl + 1) + next_t
    return next_s, reward, (next_t >= tl)

@njit(cache=True)
def sysadmin_ring_rollout(s, nc, tl, probs, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, nc + 1)
        curr_s, r, done = sysadmin_ring_step(curr_s, a, nc, tl, probs)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward
# --- Frozen Lake ---

@njit(cache=True)
def frozen_lake_step(s, a, grid, w, h, slip):
    # s is the index in the flattened grid
    x, y = s % w, s // w
    
    # Outcomes: [a-1, a, a+1] if slip, else [a]
    if slip:
        p = np.random.random()
        if p < 1.0/3.0: eff_a = (a - 1) % 4
        elif p < 2.0/3.0: eff_a = a
        else: eff_a = (a + 1) % 4
    else:
        eff_a = a
        
    dx, dy = 0, 0
    if eff_a == 0: dx = -1 # LEFT
    elif eff_a == 1: dy = 1 # DOWN
    elif eff_a == 2: dx = 1 # RIGHT
    elif eff_a == 3: dy = -1 # UP
    
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        nx, ny = x, y
        
    next_s = ny * w + nx
    cell = grid[next_s] # 0=F, 1=H, 2=G, 3=S
    
    reward = 1.0 if cell == 2 else 0.0
    done = (cell == 1 or cell == 2)
    return next_s, reward, done

@njit(cache=True)
def frozen_lake_rollout(s, grid, w, h, slip, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 4)
        curr_s, r, done = frozen_lake_step(curr_s, a, grid, w, h, slip)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward

# --- Taxi ---

@njit(cache=True)
def taxi_step(s, a, rainy_prob):
    # s: ((taxi_row * 5 + taxi_col) * 5 + pass_idx) * 4 + dest_idx
    # a: 0=S, 1=N, 2=E, 3=W, 4=Pickup, 5=Dropoff
    
    dest_idx = s % 4
    s //= 4
    pass_idx = s % 5
    s //= 5
    taxi_col = s % 5
    taxi_row = s // 5
    
    locs = np.array([(0, 0), (0, 4), (4, 0), (4, 4)])
    
    reward = -1.0
    done = False
    
    # Rainy stochasticity: move fails with prob rainy_prob
    if a < 4 and np.random.random() < rainy_prob:
        # Move fails, stay in place
        pass
    else:
        if a == 0: # SOUTH
            taxi_row = min(taxi_row + 1, 4)
        elif a == 1: # NORTH
            taxi_row = max(taxi_row - 1, 0)
        elif a == 2: # EAST
            # Walls at (row, col): (0,1), (0,3), (1,1), (1,3), (2,1), (2,3), (3,0), (3,2), (4,0), (4,2)
            if not ((taxi_row < 3 and (taxi_col == 1 or taxi_col == 3)) or 
                    (taxi_row >= 3 and (taxi_col == 0 or taxi_col == 2))):
                taxi_col = min(taxi_col + 1, 4)
        elif a == 3: # WEST
            if not ((taxi_row < 3 and (taxi_col == 2 or taxi_col == 4)) or 
                    (taxi_row >= 3 and (taxi_col == 1 or taxi_col == 3))):
                taxi_col = max(taxi_col - 1, 0)
        elif a == 4: # PICKUP
            if pass_idx < 4 and taxi_row == locs[pass_idx, 0] and taxi_col == locs[pass_idx, 1]:
                pass_idx = 4
            else:
                reward = -10.0
        elif a == 5: # DROPOFF
            if pass_idx == 4 and taxi_row == locs[dest_idx, 0] and taxi_col == locs[dest_idx, 1]:
                pass_idx = dest_idx
                reward = 20.0
                done = True
            else:
                reward = -10.0
                
    next_s = ((taxi_row * 5 + taxi_col) * 5 + pass_idx) * 4 + dest_idx
    return next_s, reward, done

@njit(cache=True)
def taxi_rollout(s, rainy_prob, limit, gamma):
    total_reward = 0.0
    disc = 1.0
    curr_s = s
    for _ in range(limit):
        a = np.random.randint(0, 6)
        curr_s, r, done = taxi_step(curr_s, a, rainy_prob)
        total_reward += disc * r
        disc *= gamma
        if done:
            break
    return total_reward
