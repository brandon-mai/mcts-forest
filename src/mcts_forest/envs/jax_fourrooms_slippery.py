from typing import Any
import jax
import jax.numpy as jnp
from gymnax.environments.misc.rooms import FourRooms, EnvState, EnvParams

class SlipperyFourRooms(FourRooms):
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        key_random, key_action = jax.random.split(key)
        # Sample whether to choose a random action
        choose_random = jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )

        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state.pos)
        reward = jnp.logical_and(
            new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        ).astype(jnp.float32)

        # Update state dict and evaluate termination conditions
        state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )