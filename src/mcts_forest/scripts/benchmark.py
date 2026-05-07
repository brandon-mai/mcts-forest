import argparse
import time
import os
import numpy as np
import pandas as pd
import random
import sys
import gymnasium as gym
import ast
import itertools
import inspect
from typing import List, Dict, Any, Tuple
import concurrent.futures
import multiprocessing

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.rule import Rule
from rich import box

from mcts_forest.utils.registry import REGISTRY
from mcts_forest.utils.experiment import generate_experiment_name, clear_directory, parse_dict
from mcts_forest.utils.stats import bootstrap_stats

console = Console()

# Configuration Defaults
DEFAULT_SOLVER_ARGS = {}
DEFAULT_ENV_ARGS = {}

def parse_grid_item(value: Any) -> List[Any]:
    """Parses a string that might be a tuple/list into a list of items."""
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

def expand_params(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expands a dictionary of parameters into all possible combinations."""
    if not params:
        return [{}]
    
    keys = list(params.keys())
    expanded_values = []
    for k in keys:
        v = params[k]
        if isinstance(v, (list, tuple)):
            expanded_values.append(list(v))
        else:
            expanded_values.append([v])
    
    combinations = list(itertools.product(*expanded_values))
    return [dict(zip(keys, combo)) for combo in combinations]

def get_solver_class(solver_name: str) -> Any:
    """Heuristic to find the actual class for a solver name to inspect its signature."""
    import mcts_forest.utils.registry as registry_mod
    mapping = {
        "uct": "UCT",
        "stochastic_uct": "StochasticUCT",
        "sp_uct": "SPUCT",
        "openloop_mcts": "OpenLoopMCTS",
        "mcgs": "MCGS",
        "gsp_uct": "GSPUCT",
        "gsp_uct_f": "GSPUCTFull",
        "gbop": "GBOP",
        "gbopd": "GBOPD",
        "ments": "MENTS",
        "gsp_alphazero": "GSPAlphaZero",
        "gsp_muzero": "GSPMuZero",
        "gsp_stochastic_muzero": "GSPStochasticMuZero",
        "alphazero": "AlphaZero",
        "muzero": "MuZero"
    }
    class_name = mapping.get(solver_name.lower())
    if class_name:
        return getattr(registry_mod, class_name, None)
    return None

def filter_compatible(solver_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filters out keys from kwargs that are incompatible with the solver."""
    solver_cls = get_solver_class(solver_name)
    if not solver_cls or not hasattr(solver_cls, '__init__'):
        return kwargs
    
    sig = inspect.signature(solver_cls.__init__)
    named_params = {p for p in sig.parameters if p not in ('self', 'env', 'kwargs', 'args')}
    universal_args = {
        'c', 'horizon', 'gamma', 'rollout_limit', 'simulation_limit', 
        'internal_reward_scale', 'internal_reward_offset', 'init_q', 
        'v_min', 'v_max', 'budget_strategy'
    }
    
    return {k: v for k, v in kwargs.items() if k in named_params or k in universal_args}

def run_episode(env_name, solver_name, sims, seed=None, episode_idx=0, solver_kwargs=None, **env_kwargs):
    """Runs a single episode and returns reward, steps, success, and mean search time."""
    solver_kwargs = solver_kwargs or {}
    env_kwargs_with_seed = env_kwargs.copy()
    if seed is not None: env_kwargs_with_seed["seed"] = seed
        
    env = REGISTRY.get_env(env_name, **env_kwargs_with_seed)
    
    reset_seed = (seed * 1000 + episode_idx) if seed is not None else None
    obs = env.reset(seed=reset_seed)

    solver = REGISTRY.get_solver(solver_name, env, simulation_limit=sims, **solver_kwargs)
    terminated, truncated = False, False
    total_reward, steps, search_times = 0, 0, []
    info = {}
    
    while not (terminated or truncated) and steps < 200:
        start_search = time.time()
        action, _ = solver.search(obs)
        search_times.append(time.time() - start_search)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    success = False
    if terminated and reward > 0:
        success = True
    elif truncated and env_name.lower().startswith("cartpole"):
        success = True
    elif terminated and "sailing" in env_name.lower():
        success = True
        
    return total_reward, steps, success, np.mean(search_times) if search_times else 0.0

def run_visualization(env_name, solver_name, sims, output_dir, target_seed, target_episode_idx, show_tree=False, solver_kwargs=None, **env_kwargs):
    from mcts_forest.utils.visualizer import print_tree
    solver_kwargs = solver_kwargs or {}
    
    env_kwargs_with_seed = env_kwargs.copy()
    env_kwargs_with_seed["seed"] = target_seed
    adapter = REGISTRY.get_env(env_name, render_mode="rgb_array", **env_kwargs_with_seed)
    
    env = gym.wrappers.RecordVideo(
        adapter.env,
        video_folder=output_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{env_name}_{solver_name}",
        disable_logger=True
    )
    
    solver = REGISTRY.get_solver(solver_name, adapter, simulation_limit=sims, **solver_kwargs)
    reset_seed = (target_seed * 1000 + target_episode_idx)
    obs, _ = env.reset(seed=reset_seed)
    
    terminated, truncated, steps = False, False, 0
    if not show_tree:
        console.print(f"[dim]Recording representative episode (Seed {target_seed}, Ep {target_episode_idx})...[/dim]")

    while not (terminated or truncated) and steps < 100:
        action, info = solver.search(obs)
        
        if show_tree and "root" in info:
            tree_log_path = os.path.join(output_dir, "tree_log.txt")
            print_tree(info["root"], path=tree_log_path, action_space_n=adapter.action_space_size, max_depth=2, c=solver.c, silent=True)
            
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        
    env.close()

def worker_fn(task):
    try:
        np.random.seed(task["seed"] * 1000 + task["episode_idx"])
        random.seed(task["seed"] * 1000 + task["episode_idx"])
        
        reward, steps, success, avg_search_time = run_episode(
            task["env_name"], task["solver_name"], task["sims"],
            seed=task["seed"],
            episode_idx=task["episode_idx"],
            solver_kwargs=task["solver_kwargs"],
            **task["env_kwargs"]
        )
        
        return {
            "seed": task["seed"],
            "episode": task["episode_idx"] % task["ep_per_seed"],
            "total_reward": reward,
            "steps": steps,
            "success": int(success),
            "avg_search_time": avg_search_time
        }
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR in worker_fn: {e}")
        traceback.print_exc()
        # Return a failure record so the main process can continue
        return {
            "seed": task["seed"],
            "episode": task["episode_idx"] % task.get("ep_per_seed", 1),
            "total_reward": 0.0,
            "steps": 0,
            "success": 0,
            "avg_search_time": 0.0,
            "error": str(e)
        }

def main():
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
    parser = argparse.ArgumentParser(description="MCTS Forest Benchmarking Suite")
    parser.add_argument("--env", type=str, default="frozenlake", help="Environment ID (Grid Search supported)")
    parser.add_argument("--solver", type=str, default="uct", help="Solver ID (Grid Search supported)")
    parser.add_argument("--sims", type=str, default="100", help="Simulations per move (Grid Search supported)")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel episode execution")
    parser.add_argument("--show_tree", action="store_true", help="Print the colorful MCTS tree to terminal during replay")
    parser.add_argument("--log", action="store_true", help="Log output files to the result directory")
    parser.add_argument("--solver_args", type=str, default="{}", help="Solver hyperparameters as dict string")
    parser.add_argument("--env_args", type=str, default="{}", help="Environment arguments as dict string")
    parser.add_argument("--table", action="store_true", help="Generate result_table.txt in markdown format")
    
    args = parser.parse_args()
    
    # 1. Parse Grid Parameters
    envs = parse_grid_item(args.env)
    solvers = parse_grid_item(args.solver)
    sims_list = sorted([int(s) for s in parse_grid_item(args.sims)])
    base_solver_args = parse_dict(args.solver_args)
    base_env_args = parse_dict(args.env_args)
    
    # 2. Generate all combinations
    all_experiments = []
    for env_name in envs:
        for solver_name in solvers:
            for sims in sims_list:
                expanded_solver_args = expand_params(base_solver_args)
                expanded_env_args = expand_params(base_env_args)
                
                unique_s_kwargs = []
                for s_kwargs in expanded_solver_args:
                    filtered = filter_compatible(solver_name, s_kwargs)
                    if filtered not in unique_s_kwargs:
                        unique_s_kwargs.append(filtered)
                
                for s_kwargs in unique_s_kwargs:
                    for e_kwargs in expanded_env_args:
                        all_experiments.append({
                            "env": env_name,
                            "solver": solver_name,
                            "sims": sims,
                            "solver_kwargs": s_kwargs,
                            "env_kwargs": e_kwargs
                        })

    if not all_experiments:
        console.print("[bold red]No compatible experiment combinations found.[/bold red]")
        return

    # Identify swept parameters for progress reporting
    swept_keys = []
    if len(envs) > 1: swept_keys.append("env")
    if len(solvers) > 1: swept_keys.append("solver")
    if len(sims_list) > 1: swept_keys.append("sims")
    
    all_s_kwargs_keys = set()
    for exp in all_experiments: all_s_kwargs_keys.update(exp["solver_kwargs"].keys())
    for k in all_s_kwargs_keys:
        if len(set(str(exp["solver_kwargs"].get(k)) for exp in all_experiments)) > 1:
            swept_keys.append(k)
            
    all_e_kwargs_keys = set()
    for exp in all_experiments: all_e_kwargs_keys.update(exp["env_kwargs"].keys())
    for k in all_e_kwargs_keys:
        if len(set(str(exp["env_kwargs"].get(k)) for exp in all_experiments)) > 1:
            swept_keys.append(k)

    # Results tracking for final table
    final_results_summary = []
    
    if any("sailing" in e.lower() for e in envs):
        # Calculate a quick baseline for Sailing (Always Down-Right)
        console.print("\n[bold blue]Sailing Baseline (Always DOWN_RIGHT):[/bold blue]")
        for env_name in envs:
            if "sailing" in env_name.lower():
                try:
                    temp_env = REGISTRY.get_env(env_name)
                    baseline_rewards = []
                    for seed in range(args.seeds):
                        for ep in range(args.episodes):
                            reset_seed = seed * 1000 + ep
                            baseline_rewards.append(temp_env.env.get_baseline_reward(reset_seed))
                    console.print(f"  • [cyan]{env_name}[/cyan]: Baseline Mean Reward = [bold]{np.mean(baseline_rewards):.2f}[/bold]")
                except:
                    pass
        console.print("")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10
    ) as progress:
        
        overall_task = progress.add_task("[bold cyan]Sweep Progress", total=len(all_experiments))
        
        for i, exp in enumerate(all_experiments):
            env_name, solver_name, sims = exp["env"], exp["solver"], exp["sims"]
            s_kwargs, e_kwargs = exp["solver_kwargs"], exp["env_kwargs"]
            
            # Descriptive label for current experiment
            label_parts = []
            for k in swept_keys:
                v = exp.get(k, s_kwargs.get(k, e_kwargs.get(k)))
                if v is not None:
                    label_parts.append(f"{k}={v}")
            current_label = " ".join(label_parts) or "Default"
            
            progress.update(overall_task, description=f"[bold cyan]Grid: {current_label} ({i+1}/{len(all_experiments)})")
            
            # Setup solver info (print once per experiment)
            with progress.console.status(f"[bold blue]Initializing {solver_name} for {env_name}...[/bold blue]"):
                temp_env = REGISTRY.get_env(env_name, **e_kwargs)
                temp_solver = REGISTRY.get_solver(solver_name, temp_env, simulation_limit=sims, **s_kwargs)
            
            if hasattr(temp_solver, "print_info"):
                temp_solver.print_info()

            # Setup output path
            output_path = None
            if args.log:
                experiment_name = generate_experiment_name(temp_env, temp_solver, seeds=args.seeds)
                output_path = os.path.join(args.output, experiment_name)
                clear_directory(output_path)
                os.makedirs(output_path, exist_ok=True)

            # Execution
            experiment_start = time.time()
            tasks = []
            for seed in range(args.seeds):
                for ep in range(args.episodes):
                    tasks.append({
                        "env_name": env_name, "solver_name": solver_name, "sims": sims,
                        "episode_idx": ep + seed * args.episodes, "seed": seed,
                        "ep_per_seed": args.episodes, "env_kwargs": e_kwargs, "solver_kwargs": s_kwargs
                    })

            exp_task = progress.add_task(f"  [dim]Episodes", total=len(tasks))
            results = []
            
            if args.parallel:
                num_workers = min(multiprocessing.cpu_count(), len(tasks))
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(worker_fn, t) for t in tasks]
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                        progress.advance(exp_task)
            else:
                for task in tasks:
                    results.append(worker_fn(task))
                    progress.advance(exp_task)
                    
            df = pd.DataFrame(results)
            progress.remove_task(exp_task)
            
            # Stats calculation
            success_mean, success_bs_std = bootstrap_stats(df["success"].values, n_resamples=2000)
            reward_mean, reward_bs_std = bootstrap_stats(df["total_reward"].values, n_resamples=2000)
            avg_steps = df['steps'].mean()
            avg_time = df['avg_search_time'].mean()
            
            # Save CSV
            if args.log and output_path:
                df.to_csv(os.path.join(output_path, "results.csv"), index=False)
                
            # Formatting for summary table
            display_name = solver_name
            if 'p' in s_kwargs:
                display_name += f" (p={s_kwargs['p']})"
            
            final_results_summary.append({
                "Solver": solver_name,
                "DisplaySolver": display_name,
                "Env": env_name,
                "Sims": sims,
                "Kwargs": str(s_kwargs) if s_kwargs else "-",
                "Success": f"{success_mean*100:.2f}% ± {success_bs_std*100:.2f}%",
                "Reward": f"{reward_mean:.4f} ± {reward_bs_std:.4f}",
                "Steps": f"{avg_steps:.2f}",
                "Time/Move": f"{avg_time:.4f}s"
            })
            
            if args.log and output_path:
                successes = df[df['success'] == 1]
                best_row = successes.sort_values('steps').iloc[0] if not successes.empty else df.sort_values('steps', ascending=False).iloc[0]
                try:
                    run_visualization(env_name, solver_name, sims, output_path, int(best_row['seed']), int(best_row['episode']), show_tree=args.show_tree, solver_kwargs=s_kwargs, **e_kwargs)
                except Exception as e:
                    console.print(f"  [yellow]⚠ Skipping video recording for {env_name}: {e}[/yellow]")

            progress.advance(overall_task)

    # 3. Final Summary Table
    console.print("\n", Rule("Benchmarking Summary", style="bold cyan"))
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_header=True, header_style="bold magenta")
    table.add_column("Solver", justify="left")
    table.add_column("Env", justify="left")
    table.add_column("Sims", justify="right")
    table.add_column("Solver Args", justify="left", overflow="fold")
    table.add_column("Success Rate", justify="right", no_wrap=True)
    table.add_column("Mean Reward", justify="right", no_wrap=True)
    table.add_column("Steps", justify="right")
    table.add_column("Time/Move", justify="right")

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
            res["Time/Move"]
        )

    console.print(table)
    
    # Generate Markdown Table if requested
    if args.table:
        # Pivot results: rows = DisplaySolver, columns = Sims
        # We assume one Env for simplicity in the text table as per user example
        unique_display_solvers = []
        for res in final_results_summary:
            if res["DisplaySolver"] not in unique_display_solvers:
                unique_display_solvers.append(res["DisplaySolver"])
        
        sims_list = sorted(list(set(res["Sims"] for res in final_results_summary)))
        
        md_lines = []
        # Header
        header = "| | " + " | ".join(str(s) for s in sims_list) + " |"
        separator = "| --- | " + " | ".join("---" for _ in sims_list) + " |"
        md_lines.append(header)
        md_lines.append(separator)
        
        for ds in unique_display_solvers:
            row_parts = [ds]
            for s in sims_list:
                # Find matching result
                match = next((res for res in final_results_summary if res["DisplaySolver"] == ds and res["Sims"] == s), None)
                if match:
                    cell = f"{match['Success']}<br>**{match['Reward']}**"
                    row_parts.append(cell)
                else:
                    row_parts.append("")
            md_lines.append("| " + " | ".join(row_parts) + " |")
        
        with open("result_table.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        console.print(f"[bold green]Markdown table saved to result_table.txt[/bold green]")

    elapsed = progress.tasks[0].elapsed if 'progress' in locals() else 0
    console.print(f"[dim]All experiments completed in {elapsed:.2f} seconds.[/dim]\n")

if __name__ == "__main__":
    main()
