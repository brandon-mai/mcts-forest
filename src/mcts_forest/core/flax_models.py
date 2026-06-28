import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Any

class FourRoomsNet(nn.Module):
    num_actions: int = 4
    
    @nn.compact
    def __call__(self, x: Any) -> Any:
        # Check if input is batched. If not, add batch dimension.
        is_batched = x.ndim > 1
        if not is_batched:
            x = x[None, :]
            
        agent_x = x[:, 0].astype(jnp.int32)
        agent_y = x[:, 1].astype(jnp.int32)
        goal_x = x[:, 2].astype(jnp.int32)
        goal_y = x[:, 3].astype(jnp.int32)
        
        B = x.shape[0]
        
        agent_x_oh = jax.nn.one_hot(agent_x, 13)
        agent_y_oh = jax.nn.one_hot(agent_y, 13)
        agent_grid = jnp.einsum('bi,bj->bij', agent_x_oh, agent_y_oh)
        
        goal_x_oh = jax.nn.one_hot(goal_x, 13)
        goal_y_oh = jax.nn.one_hot(goal_y, 13)
        goal_grid = jnp.einsum('bi,bj->bij', goal_x_oh, goal_y_oh)
        
        wall_map = jnp.array([
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=jnp.float32)
        
        wall_grid = jnp.broadcast_to(wall_map[None, :, :], (B, 13, 13))
        grid = jnp.stack([wall_grid, agent_grid, goal_grid], axis=-1) # [B, 13, 13, 3]
        
        # Conv backbone
        x_conv = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(grid)
        x_conv = nn.relu(x_conv)
        x_conv = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x_conv)
        x_conv = nn.relu(x_conv)
        
        # Flatten
        x_flat = x_conv.reshape((B, -1))
        
        # Dense layers
        x_dense = nn.Dense(128)(x_flat)
        x_dense = nn.relu(x_dense)
        
        logits = nn.Dense(self.num_actions)(x_dense)
        value = nn.Dense(1)(x_dense)
        
        # Squeeze batch if input was not batched
        if not is_batched:
            logits = jnp.squeeze(logits, axis=0)
            value = jnp.squeeze(value, axis=0)
            
        return logits, jnp.squeeze(value, axis=-1)

class Game2048Net(nn.Module):
    num_actions: int = 4
    
    @nn.compact
    def __call__(self, x: Any) -> Any:
        # Check if x is the Jumanji observation NamedTuple
        if hasattr(x, "board"):
            board = x.board
        else:
            board = x
            
        # board shape: [B, 4, 4] or [4, 4] (if vmapped)
        is_batched = board.ndim > 2
        if not is_batched:
            board = board[None, :, :]
            
        B = board.shape[0]
        
        # Normalize: divide exponents by 16.0 to range [0, 1]
        board_norm = board.astype(jnp.float32) / 16.0
        
        # Add channel dimension: [B, 4, 4, 1]
        grid = jnp.expand_dims(board_norm, axis=-1)
        
        # Conv backbone
        x_conv = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(grid)
        x_conv = nn.relu(x_conv)
        x_conv = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x_conv)
        x_conv = nn.relu(x_conv)
        
        # Flatten
        x_flat = x_conv.reshape((B, -1))
        
        # Dense layers
        x_dense = nn.Dense(128)(x_flat)
        x_dense = nn.relu(x_dense)
        
        logits = nn.Dense(self.num_actions)(x_dense)
        value = nn.Dense(1)(x_dense)
        
        # Squeeze batch if input was not batched
        if not is_batched:
            logits = jnp.squeeze(logits, axis=0)
            value = jnp.squeeze(value, axis=0)
            
        return logits, jnp.squeeze(value, axis=-1)
