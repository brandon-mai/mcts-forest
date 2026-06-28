# Monte Carlo Forest 🌲

## Development Strategy
- **Surgical Changes**: We prioritize stability. Changes to the core math (like the Dynamic Power-Mean Shift) are ported across all solvers simultaneously to maintain parity.
- **Performance**: We use Numba for tree operations and PyTorch for batch training.
- **Protocol-Based Interface**: Lean `GymAdapter` that supports state-snapshotting for Gymnasium environments without redundant abstractions.
- **Dynamic Registry**: Centralized `REGISTRY` for managing solvers (UCT, MCGS) and environments (FrozenLake, Taxi).
- **Rigorous Benchmarking**: Support for multi-seed experiments, CSV data export, and automated summary statistics.

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Usage

### Benchmarking

Run the benchmarking script to evaluate a solver on an environment. The script supports **Grid Search** for comparing multiple configurations in a single run.

### Example Commands

#### 1. Frozen Lake (Slippery)
```bash
# Debugging
uv run benchmark --env frozenlake_slip --solver uct --sims 10 --seeds 1

# Single Solver Analysis
uv run benchmark --env frozenlake_slip --solver gbop --sims "(1024, 2048, 4096, 8192, 16384)" --seeds 1000 --parallel

# Comprehensive Benchmark
uv run benchmark --env frozenlake_slip --solver "('stochastic_uct', 'sp_uct', 'gbopd', 'gbop', 'gsp_uct')" --sims "(1024, 2048, 4096, 8192, 16384)" --seeds 1000 --solver_args "{'p': (1.0, 2.0)}" --parallel --table
```

#### 2. Taxi (Rainy)
```bash
# Debugging
uv run benchmark --env taxi_rain --solver uct --sims 50 --seeds 1 --solver_args "{'internal_reward_scale': 0.0333, 'c': 2.0}"

# Single Solver Analysis
uv run benchmark --env taxi_rain --solver uct --sims "(512, 1024, 2048, 4096, 8192)" --seeds 50 --solver_args "{'internal_reward_scale': 0.0333, 'c': 2.0, 'init_q': -100, 'v_min': -6.6, 'v_max': 0.66}" --parallel

# Comprehensive Benchmark
uv run benchmark --env taxi_rain --solver "('stochastic_uct', 'sp_uct', 'ments', 'gbopd', 'gbop', 'gsp_uct')" --sims "(512, 1024, 2048, 4096, 8192)" --seeds 50 --solver_args "{'p': (1.0, 2.0), 'internal_reward_scale': 0.0333, 'c': 2.0, 'init_q': -100, 'v_min': -6.6, 'v_max': 0.66}" --parallel --table
```

#### 3. Sailing (Windy)
```bash
# Debugging
uv run benchmark --env sailing8x8_0.8 --solver uct --sims 100 --seeds 1 --solver_args "{'internal_reward_scale': 0.5}" --parallel

# Single Solver Analysis
uv run benchmark --env sailing8x8_0.8 --solver uct --sims "(512, 1024, 2048, 4096, 8192)" --seeds 50 --solver_args "{'internal_reward_scale': 0.5}" --parallel

# Comprehensive Benchmark
uv run benchmark --env sailing8x8_0.8 --solver "('stochastic_uct', 'sp_uct', 'ments', 'gbopd', 'gbop', 'gsp_uct')" --sims "(128, 256, 512, 1024, 2048)" --seeds 100 --solver_args "{'p': (1.0, 2.0), 'internal_reward_scale': 0.5}" --parallel --table
```

#### 4. JAIR Benchmarks (Factored River Swim, Four Rooms, Passenger Grid, Sysadmin)
```bash
# Passenger Grid (Sims: 256 to 8192)
uv run benchmark --env passenger_grid --solver gbop --sims "(1024)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 70}"

uv run benchmark --env passenger_grid --solver gbop --sims "(256, 512, 1024, 2048, 4096, 8192)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 70}" --table

# Factored River Swim (Sims: 64 to 2048)
uv run benchmark --env riverswim_n4x8 --solver gbop --sims "(1024)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 35}"

uv run benchmark --env riverswim_n4x8 --solver gbop --sims "(64, 128, 256, 512, 1024, 2048)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 35}" --table

# Sysadmin Ring (Sims: 512 to 16384)
uv run benchmark --env sysadmin_n20 --solver gbop --sims "(512, 1024, 2048, 4096, 8192, 16384)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 50}" --table

# Four Rooms (Sims: 64 to 4096)
uv run benchmark --env fourrooms_n5 --solver gbop --sims "(64, 128, 256, 512, 1024, 2048, 4096)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 50}" --table
```

### Arguments

- `--env`: Environment ID (e.g., `frozenlake`, `taxi`). Supports Grid Search `"(env1, env2)"`.
- `--solver`: Solver ID (e.g., `uct`, `sp_uct`). Supports Grid Search `"(s1, s2)"`.
- `--sims`: MCTS simulations per move. Supports Grid Search `"(100, 200)"`.
- `--solver_args`: Dictionary string for solver hypers. Values can be grids `"{'p': (1.0, 2.0)}"`.
- `--log`: Boolean flag. If present, saves CSVs, videos, and logs to `results/`.
- `--episodes`: Episodes per seed (default: 1).
- `--seeds`: Number of random seeds for statistical significance (default: 5).
- `--parallel`: Enable parallel episode execution.
- `--show_tree`: Print the colorful MCTS tree to terminal during replay.
- `--table`: Generate `result_table.txt` with a Markdown summary table.

### Benchmaxxing

Benchmark with JAX (!!!)
```bash
# Jumanji 2048
uv run benchmark --env 2048 --solver mctx --sims 100 --seeds 1000

# Gymnax Four Rooms
uv run benchmark --env fourrooms --solver mctx --sims 100 --seeds 1000

# UCT
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard', 'p': 1.0, 'gamma': 0.95}" --table

uv run benchmark --env 2048 --solver mctx --sims "(16, 32, 64, 128)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard', 'p': 1.0, 'gamma': 0.95, 'num_chance_outcomes': 30}" --table

# SPUCT
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 2.0, 'gamma': 0.95}" --table

# GSPUCT
uv run benchmark --env 2048 --solver mctx --sims "(128)" --seeds 1000 --solver_args "{'merge_mode': 'depth_dependent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': (1.0, 1.2, 1.5, 2.0, 5.0, 10.0, 'inf'), 'gamma': 0.95}" --table

# GSPUCTF
uv run benchmark --env fourrooms --solver mctx --sims "(128)" --seeds 1000 --solver_args "{'merge_mode': 'depth_independent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.0, 'gamma': 0.95}" --table

# ULTIMATE FOUR ROOMS
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': ('depth_dependent', 'depth_independent'), 'max_depth': 3, 'ucb_mode': 'spuct', 'p': (1.0, 2.0), 'gamma': 0.95}" --table

# ULTIMATE 2048
uv run benchmark --env 2048 --solver mctx --sims "(32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': ('pure_tree', 'depth_dependent', 'depth_independent'), 'max_depth': 3, 'ucb_mode': 'spuct', 'p': (1.0, 2.0), 'gamma': 0.95, 'num_chance_outcomes': 30}" --table
```

### AlphaZero Training

You can train dual-headed Flax actor-critic networks using parallelized JAX self-play generated by either the old MCTS (UCT) or the new MCTS (GSPUCT).

#### Production Training Commands

To obtain high-quality trained networks, use the following serious training commands:

```bash
# 1. Train using UCT (Old MCTS: no state merging, standard UCB, infinite horizon)
uv run train --env fourrooms --sims 128 --parallel_envs 128 --max_steps 50000 --batch_size 1024 --buffer_size 50000 --learning_rate 1e-4 --save_freq 1000 --eval_freq 200 --checkpoint_dir "results/uct_fourrooms" --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard'}"

# 1.1. Benchmark with trained network
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128)" --seeds 1000 --model_checkpoint results/uct_fourrooms/checkpoint_latest.msgpack --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard'}"

# 2. Train using GSPUCTF (New MCTS: depth-independent state merging, SP-UCT UCB, limited horizon)
uv run train --env fourrooms --sims 128 --parallel_envs 32 --max_steps 50000 --batch_size 128 --buffer_size 50000 --learning_rate 5e-4 --save_freq 1000 --checkpoint_dir "results/gspuct_fourrooms" --solver_args "{'merge_mode': 'depth_independent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.2}"
```

#### Training Arguments

- `--env`: Environment to train on (`fourrooms` or `2048`).
- `--sims`: MCTS simulations per step (budget) during self-play. Higher values provide stronger training targets but increase training time.
- `--parallel_envs`: Number of parallel environments running concurrently on GPU/device to generate trajectories. Adjust based on GPU memory.
- `--max_steps`: Total training steps (number of gradient updates).
- `--batch_size`: Mini-batch size sampled from the replay buffer for each training step.
- `--buffer_size`: Maximum transition capacity of the circular replay buffer.
- `--learning_rate`: Step-size for Optax Adam optimizer.
- `--eval_freq`: Frequency of logging metrics to the terminal (in steps).
- `--max_depth`: Default search max depth (horizon) if not overwritten by `solver_args`.
- `--gamma`: Discount factor for returns and transitions.
- `--solver_args`: Dictionary specifying search settings (`merge_mode`, `max_depth`, `ucb_mode`, `p`) used during self-play.
- `--checkpoint_dir`: Path to folder where training weights are serialized.
- `--save_freq`: Frequency of writing checkpoints to disk (in steps).
- `--resume`: If present, automatically resumes training from the latest checkpoint (`checkpoint_latest.msgpack`) found in `--checkpoint_dir`.

#### Monitoring & Tuning Guide

During execution, watch the terminal logging bar:
- **Loss**: Policy cross-entropy + value MSE. Should steadily decrease. If it diverges or fluctuates wildly, reduce `--learning_rate` (e.g. to `1e-4` or `5e-5`).
- **Mean Reward**: Average unnormalized raw game score achieved by current self-play. Should steadily increase as the policy improves.
- **Mean Ep Len**: Average episode steps. In 2048, a higher step length indicates survival and better play.
- **Policy Collapse Prevention**: If performance degrades suddenly, stop and restart from a previous stable checkpoint using `checkpoint_<step>.msgpack` renamed as `checkpoint_latest.msgpack` and the `--resume` flag.

## Visualization & Analysis

### Plots & CSVs
Aggregated results are stored in `results/<experiment_name>/`:
- `results.csv`: Raw data (seed, episode, reward, steps).
- `bench_plot.png`: Rolling average reward visualization.

## Project Structure

- `core/`: MCTS logic, variants (UCT), and baselines.
- `envs/`: `GymAdapter` and Gymnasium integration.
- `utils/`: Logging and helper utilities.
- `results/`: [Internal] Output data, plots.
- `scripts/`: Entry points for benchmarking and tuning.

## License

MIT
