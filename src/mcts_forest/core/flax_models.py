import flax.linen as nn
import jax.numpy as jnp
from typing import Any

class FourRoomsNet(nn.Module):
    num_actions: int = 4
    
    @nn.compact
    def __call__(self, x: Any) -> Any:
        # x is expected to be a JAX array of shape [B, 3] or similar
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x)
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
        if board.ndim == 2:
            flat_board = board.reshape(16).astype(jnp.float32)
        else:
            flat_board = board.reshape((board.shape[0], 16)).astype(jnp.float32)
        
        x = nn.Dense(256)(flat_board)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x)
        return logits, jnp.squeeze(value, axis=-1)
