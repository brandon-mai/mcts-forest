import argparse
import time
import jax
import jax.numpy as jnp
import optax
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from mcts_forest.scripts.jax_benchmark import make_gymnax_fns, make_jumanji_fns
from mcts_forest.utils.experiment import parse_dict
from mcts_forest.core.flax_models import FourRoomsNet, Game2048Net
from mcts_forest.core.jax_alphazero import (
    init_replay_buffer,
    add_to_buffer,
    sample_buffer,
    self_play_episode_batch,
    train_step
)
import os
from flax import serialization

console = Console()

def save_checkpoint(checkpoint_dir, params, step):
    os.makedirs(checkpoint_dir, exist_ok=True)
    bytes_data = serialization.to_bytes(params)
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.msgpack")
    with open(path, "wb") as f:
        f.write(bytes_data)
    # Save latest copy
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.msgpack")
    with open(latest_path, "wb") as f:
        f.write(bytes_data)
    console.print(f"[bold green]Checkpoint saved to {path}[/bold green]")

def main():
    parser = argparse.ArgumentParser(description="AlphaZero JAX Training Pipeline")
    parser.add_argument("--env", type=str, default="fourrooms", choices=["fourrooms", "2048"], help="Environment name")
    parser.add_argument("--sims", type=int, default=10, help="Number of MCTS simulations per step")
    parser.add_argument("--parallel_envs", type=int, default=16, help="Number of parallel self-play environments")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Maximum number of transitions in replay buffer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation/Log frequency (iterations)")
    parser.add_argument("--max_depth", type=int, default=10, help="Search max depth")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--solver_args", type=str, default="{}", help="Solver hyperparameters as dict string")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if available")
    parser.add_argument("--save_freq", type=int, default=100, help="Checkpoint saving frequency (steps)")
    args = parser.parse_args()

    console.print(f"[bold cyan]Initializing {args.env} environment...[/bold cyan]")

    solver_kwargs = parse_dict(args.solver_args)

    if args.env == "fourrooms":
        reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions = make_gymnax_fns("FourRooms-misc")
        model = FourRoomsNet(num_actions=num_actions)
        max_episode_steps = 250
    else:
        reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions = make_jumanji_fns("Game2048-v1")
        model = Game2048Net(num_actions=num_actions)
        max_episode_steps = 500

    # Initialize model variables
    key = jax.random.PRNGKey(0)
    key, init_key, run_key = jax.random.split(key, 3)
    
    # Run a dummy reset to get observation structure
    dummy_key = jax.random.PRNGKey(42)
    _, dummy_obs = reset_fn(dummy_key)
    
    # Expand to batch dimension for init
    dummy_batched_obs = jax.tree.map(lambda x: jnp.expand_dims(x, 0), dummy_obs)
    
    console.print("[bold cyan]Initializing Flax Neural Network...[/bold cyan]")
    params = model.init(init_key, dummy_batched_obs)
    
    # Resume from checkpoint if requested and exists
    latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.msgpack")
    if args.resume and os.path.exists(latest_path):
        console.print(f"[bold green]Resuming training. Loading checkpoint: {latest_path}[/bold green]")
        with open(latest_path, "rb") as f:
            bytes_data = f.read()
        params = serialization.from_bytes(params, bytes_data)

    # Initialize Optimizer
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize Replay Buffer
    console.print("[bold cyan]Initializing Replay Buffer...[/bold cyan]")
    buffer = init_replay_buffer(args.buffer_size, dummy_obs, num_actions)
    
    # Fill replay buffer with initial self-play runs (warmup)
    console.print("[bold yellow]Generating initial warm-up trajectories...[/bold yellow]")
    warmup_key, run_key = jax.random.split(run_key)
    max_depth_val = solver_kwargs.pop("max_depth", args.max_depth)
    trajectory = self_play_episode_batch(
        warmup_key,
        params,
        model,
        reset_fn,
        step_fn,
        action_mask_fn,
        reward_norm_fn,
        state_equal_fn,
        num_actions,
        args.parallel_envs,
        max_episode_steps,
        args.sims,
        max_depth_val,
        args.gamma,
        **solver_kwargs
    )
    buffer = add_to_buffer(
        buffer,
        trajectory["obs"],
        trajectory["target_policy"],
        trajectory["target_value"],
        trajectory["active_mask"]
    )
    
    console.print(f"[bold green]Replay buffer warmed up with {buffer.current_size} transitions.[/bold green]")
    
    # Main training loop
    n_iterations = args.max_steps // args.eval_freq
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Training AlphaZero...", total=n_iterations)
        
        for iteration in range(n_iterations):
            run_key, play_key, train_key = jax.random.split(run_key, 3)
            
            # 1. Self-Play step to generate more experience
            trajectory = self_play_episode_batch(
                play_key,
                params,
                model,
                reset_fn,
                step_fn,
                action_mask_fn,
                reward_norm_fn,
                state_equal_fn,
                num_actions,
                args.parallel_envs,
                max_episode_steps,
                args.sims,
                max_depth_val,
                args.gamma,
                **solver_kwargs
            )
            buffer = add_to_buffer(
                buffer,
                trajectory["obs"],
                trajectory["target_policy"],
                trajectory["target_value"],
                trajectory["active_mask"]
            )
            
            # Calculate metrics
            mean_reward = jnp.sum(trajectory["reward"]) / args.parallel_envs
            mean_episode_length = jnp.sum(trajectory["active_mask"]) / args.parallel_envs
            
            # 2. Optimization updates
            losses = []
            train_keys = jax.random.split(train_key, args.eval_freq)
            for step in range(args.eval_freq):
                batch = sample_buffer(train_keys[step], buffer, args.batch_size)
                params, opt_state, loss = train_step(
                    params,
                    opt_state,
                    model,
                    optimizer,
                    batch
                )
                losses.append(loss)
                
            avg_loss = jnp.mean(jnp.array(losses))
            
            # Logs
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Train Step { (iteration+1)*args.eval_freq }/{args.max_steps} | Loss: {avg_loss:.4f} | Mean Reward: {mean_reward:.2f} | Mean Ep Len: {mean_episode_length:.1f}"
            )
            
            # Save checkpoint periodically
            step_num = (iteration + 1) * args.eval_freq
            if step_num % args.save_freq == 0:
                save_checkpoint(args.checkpoint_dir, params, step_num)
                
    console.print("[bold green]Training complete![/bold green]")

if __name__ == "__main__":
    main()
