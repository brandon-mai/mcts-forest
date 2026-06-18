## Stochastic Power UCT implementation

### Global data structure (either global or passed around as function arguments)

- List of V-nodes (state), each node stores:
    - Value (V)
    - Visits (T_s)

- List of Q-nodes (state-action), each node stores:
    - Value (Q)
    - Visits (T_sa)

- one V-node has many children of Q-nodes (one state - many actions)
- one Q-node may have many children of V-nodes (depends on transition probability distribution - stochasticity)

### Functions

- rollout(state):
    - Return estimated V of a state (value of V-node of state)

- select_action(state, greedy):
    - If greedy: return action from Q-node list so that (Q-value + exploration term) is maximum
    - Else: return action from Q-node list so that (Q-value) is maximum
    - exploration term is (C * (T_s ** 1/4) / (T_sa ** 1/2))

- simulate_V(state, depth):
    - action = select_action(state, greedy = false)
    - simulate_Q(state, action, depth)
    - T_s = T_s + 1
    - new V-value of state = ((T_sa / T_s) * (Q-value ** p) summed for all actions (children) from state) ** (1/p)

- simulate_Q(state, action, depth):
    - next_state = transition(state, action)
    - reward = reward(state, action, next_state)
    - If next_state is not terminal (depth + 1 != max depth H):
        - if next_state is not expanded: V-value of next_state is rollout(next_state)
        - else: simulate_V(next_state, depth + 1)
    - Update:
        - new Q-value of state = (old Q-value of state * T_sa + reward + gamma * V-value of next_state) / (T_sa + 1)
        - T_sa = T_sa + 1

- main(): at each decision making step:
    - reset all nodes, start fresh
    - for _ in number_of_simulations:
        - simulate_V(state, depth = 0)
    - return action = select_action(state, greedy = true) from final tree