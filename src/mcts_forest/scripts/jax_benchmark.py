import argparse
import time
import os
import numpy as np
import pandas as pd
import sys
import ast
import itertools
from typing import List, Dict, Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import gymnax
import jumanji

# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_dump_ir_to", "jax_ir")
# jax.config.update("jax_dump_ir_modes", "eqn_count_pprof")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box
from rich.rule import Rule

from mcts_forest.utils.stats import bootstrap_stats
from mcts_forest.utils.experiment import parse_dict
from mcts_forest.core.jax_random import jax_random_search
from mcts_forest.core.jax_spuct import jax_spuct_search
from mcts_forest.core.jax_mctxf import jax_mctx_search

console = Console()

# 1. Normalized functional environment interfaces

def make_gymnax_fns(env_name: str):
    env, params = gymnax.make(env_name)
    
    def reset_fn(key):
        obs, state = env.reset(key, params)
        return state, obs
        
    def step_fn(key, state, action):
        obs, next_state, reward, done, info = env.step(key, state, action, params)
        return next_state, obs, reward, done, info
        
    action_mask_fn = lambda obs: jnp.ones(int(env.num_actions), dtype=jnp.bool_)
    reward_norm_fn = lambda r: r
    state_equal_fn = lambda s1, s2: jnp.all(s1.pos == s2.pos)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, int(env.num_actions)

def make_jumanji_fns(env_name: str):
    env = jumanji.make(env_name)
    
    def reset_fn(key):
        state, timestep = env.reset(key)
        return state, timestep.observation
        
    def step_fn(key, state, action):
        next_state, timestep = env.step(state, action)
        done = timestep.last()
        reward = timestep.reward
        return next_state, timestep.observation, reward, done, {}
        
    action_mask_fn = lambda obs: obs.action_mask
    reward_norm_fn = lambda r: jnp.log2(r + 1.0) / jnp.log2(131072.0)
    state_equal_fn = lambda s1, s2: jnp.all(s1.board == s2.board)
    return reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, int(env.action_spec.num_values)

# 2. Vectorized simulation scan

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def run_episodes_jax(key, reset_fn, step_fn, solver_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions, num_seeds, max_steps):
    init_key, loop_key = jax.random.split(key)
    seed_keys = jax.random.split(init_key, num_seeds)
    
    # Parallel reset
    states, obss = jax.vmap(reset_fn)(seed_keys)
    
    active_mask = jnp.ones(num_seeds, dtype=jnp.bool_)
    total_rewards = jnp.zeros(num_seeds, dtype=jnp.float32)
    episode_steps = jnp.zeros(num_seeds, dtype=jnp.int32)
    success_mask = jnp.zeros(num_seeds, dtype=jnp.bool_)
    total_node_counts = jnp.zeros(num_seeds, dtype=jnp.float32)
    
    carry = (states, obss, loop_key, active_mask, total_rewards, episode_steps, success_mask, total_node_counts)
    
    def scan_body(carry, _):
        states, obss, key, active_mask, total_rewards, episode_steps, success_mask, total_node_counts = carry
        
        # Split key for parallel search and step keys
        key, step_key = jax.random.split(key)
        subkeys = jax.random.split(step_key, 2 * num_seeds)
        search_keys = subkeys[:num_seeds]
        env_keys = subkeys[num_seeds:]
        
        # Parallel solver search - returns action and node_count
        actions, node_counts = jax.vmap(solver_fn, in_axes=(0, 0, 0))(
            search_keys, obss, states
        )
        
        # Parallel environment step
        next_states, next_obss, rewards, dones, _ = jax.vmap(step_fn)(env_keys, states, actions)
        
        # Update metrics
        new_total_rewards = total_rewards + rewards * active_mask
        new_episode_steps = episode_steps + active_mask.astype(jnp.int32)
        new_success_mask = success_mask | (active_mask & dones & (rewards > 0.0))
        new_active_mask = active_mask & ~dones
        new_total_node_counts = total_node_counts + node_counts * active_mask
        
        next_carry = (next_states, next_obss, key, new_active_mask, new_total_rewards, new_episode_steps, new_success_mask, new_total_node_counts)
        return next_carry, None

    final_carry, _ = jax.lax.scan(scan_body, carry, xs=None, length=max_steps)
    _, _, _, _, final_rewards, final_steps, final_success, final_node_counts = final_carry
    
    return final_rewards, final_steps, final_success, final_node_counts

def parse_grid_item(value: Any) -> List[Any]:
    if isinstance(value, str):
        value = value.strip()
        if (value.startswith('(') and value.endswith(')')) or (value.startswith('[') and value.endswith(']')):
            try:
                res = ast.literal_eval(value)
                if isinstance(res, (list, tuple)):
                    return list(res)
                return [res]
            except:
                return [value]
    return [value]

def main():
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="JAX-Native Benchmarking Suite (benchmax)")
    parser.add_argument("--env", type=str, default="2048", help="Environment ID (fourrooms, 2048)")
    parser.add_argument("--solver", type=str, default="random", help="Solver ID")
    parser.add_argument("--sims", type=str, default="100", help="Simulations per move")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, default=1000, help="Number of random seeds (vectorized batch size)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--table", action="store_true", help="Generate result_table.txt in markdown format")
    parser.add_argument("--solver_args", type=str, default="{}", help="Solver hyperparameters as dict string")
    
    args = parser.parse_args()
    
    envs = parse_grid_item(args.env)
    solvers = parse_grid_item(args.solver)
    sims_list = sorted([int(s) for s in parse_grid_item(args.sims)])
    solver_kwargs = parse_dict(args.solver_args)
    
    final_results_summary = []
    
    num_seeds = args.seeds * args.episodes
    max_steps = 500
    
    console.print(f"[bold cyan]Starting benchmax with batch size {num_seeds}...[/bold cyan]")
    
    for env_name in envs:
        # Load environment fns
        if "fourrooms" in env_name.lower():
            reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions = make_gymnax_fns("FourRooms-misc")
            env_disp = "fourrooms"
        elif "2048" in env_name.lower():
            reset_fn, step_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions = make_jumanji_fns("Game2048-v1")
            env_disp = "2048"
        else:
            console.print(f"[bold red]Unsupported environment: {env_name}[/bold red]")
            continue
            
        for solver_name in solvers:
            if solver_name.lower() not in ["random", "spuct", "mctx"]:
                console.print(f"[bold red]Unsupported solver: {solver_name}[/bold red]")
                continue
                
            for sims in sims_list:
                if solver_name.lower() == "random":
                    def solver_fn(k, o, s):
                        act = jax_random_search(k, o, s, step_fn, reset_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions)
                        return act, 0.0
                elif solver_name.lower() == "mctx":
                    def solver_fn(k, o, s):
                        act, node_count = jax_mctx_search(k, o, s, step_fn, reset_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions, num_simulations=sims, return_node_count=True, **solver_kwargs)
                        return act, node_count
                else:
                    def solver_fn(k, o, s):
                        act = jax_spuct_search(k, o, s, step_fn, reset_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions, num_simulations=sims, **solver_kwargs)
                        return act, float(sims + 2)
                    
                with console.status(f"[bold blue]Compiling JAX graph for {env_disp}...[/bold blue]"):
                    # Warmup run to compile
                    warmup_key = jax.random.PRNGKey(0)
                    _ = run_episodes_jax(warmup_key, reset_fn, step_fn, solver_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions, num_seeds, max_steps)
                
                with console.status(f"[bold green]Running {num_seeds} vectorized episodes on device...[/bold green]"):
                    key = jax.random.PRNGKey(42)
                    t0 = time.time()
                    rewards, steps, success, node_counts = run_episodes_jax(
                        key, reset_fn, step_fn, solver_fn, action_mask_fn, reward_norm_fn, state_equal_fn, num_actions, num_seeds, max_steps
                    )
                    # Block to sync device
                    steps.block_until_ready()
                    elapsed = time.time() - t0
                
                # Retrieve arrays to host
                rewards_np = np.array(rewards)
                steps_np = np.array(steps)
                success_np = np.array(success).astype(np.int32)
                node_counts_np = np.array(node_counts)
                
                success_mean, success_bs_std = bootstrap_stats(success_np, n_resamples=2000)
                reward_mean, reward_bs_std = bootstrap_stats(rewards_np, n_resamples=2000)
                avg_steps = steps_np.mean()
                total_steps = int(np.sum(steps_np))
                avg_time = elapsed / total_steps if total_steps > 0 else 0.0
                avg_nodes = np.mean(node_counts_np / np.maximum(steps_np, 1))
                
                final_results_summary.append({
                    "Solver": solver_name,
                    "DisplaySolver": solver_name,
                    "Env": env_disp,
                    "Sims": sims,
                    "Kwargs": str(solver_kwargs) if solver_kwargs else "-",
                    "Success": f"{success_mean*100:.2f}% ± {success_bs_std*100:.2f}%",
                    "Reward": f"{reward_mean:.4f} ± {reward_bs_std:.4f}",
                    "Steps": f"{avg_steps:.2f}",
                    "Time/Move": f"{avg_time*1000:.4f}ms",
                    "Nodes": f"{avg_nodes:.1f}"
                })
                
    console.print("\n", Rule("benchmax Vectorized Summary", style="bold cyan"))
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_header=True, header_style="bold magenta")
    table.add_column("Solver", justify="left")
    table.add_column("Env", justify="left")
    table.add_column("Sims", justify="right")
    table.add_column("Solver Args", justify="left")
    table.add_column("Success Rate", justify="right", no_wrap=True)
    table.add_column("Mean Reward", justify="right", no_wrap=True)
    table.add_column("Steps", justify="right")
    table.add_column("Time/Move", justify="right")
    table.add_column("Avg Nodes", justify="right")
    
    for res in final_results_summary:
        color = "green" if "100.00%" in res["Success"] else "yellow" if "0.00%" not in res["Success"] else "red"
        table.add_row(
            res["DisplaySolver"],
            res["Env"],
            str(res["Sims"]),
            res["Kwargs"],
            f"[{color}]{res['Success']}[/{color}]",
            res["Reward"],
            res["Steps"],
            res["Time/Move"],
            res["Nodes"]
        )
        
    console.print(table)
    
    if args.table:
        unique_display_solvers = sorted(list(set(res["DisplaySolver"] for res in final_results_summary)))
        sims_list = sorted(list(set(res["Sims"] for res in final_results_summary)))
        
        md_lines = []
        header = "| | " + " | ".join(str(s) for s in sims_list) + " |"
        separator = "| --- | " + " | ".join("---" for _ in sims_list) + " |"
        md_lines.append(header)
        md_lines.append(separator)
        
        for ds in unique_display_solvers:
            row_parts = [ds]
            for s in sims_list:
                match = next((res for res in final_results_summary if res["DisplaySolver"] == ds and res["Sims"] == s), None)
                if match:
                    cell = f"{match['Success']}<br>**{match['Reward']}**"
                    row_parts.append(cell)
                else:
                    row_parts.append("")
            md_lines.append("| " + " | ".join(row_parts) + " |")
            
        with open("result_table.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        console.print(f"[bold green]Markdown table saved to result_table.txt[/bold green]\n")

if __name__ == "__main__":
    main()