# Monte Carlo Forest 🌲

A research-oriented framework for benchmarking and comparing Monte Carlo Tree Search (MCTS) methods with rigorous RL evaluation standards.

## Key Features

- **High-Performance Functional Core**: Numba-accelerated search engines achieving up to 1,000,000 simulations per second.
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

```bash
# Basic run (no files saved)
uv run benchmark --env frozenlake --solver uct --seeds 5 --sims 100

# Grid Search over solvers and simulations (no files saved)
uv run benchmark --env frozenlake_slip --solver "('uct', 'sp_uct')" --sims "(1024, 2048)" --seeds 100

# Grid Search with hyperparameter combinations (log results to disk)
uv run benchmark --env frozenlake_slip --solver sp_uct --solver_args "{'p': (1.0, 2.0)}" --log

# Real command I used
uv run benchmark --env frozenlake8x8_slip --solver "('mcgs', 'gsp_uct')" --seeds 1000 --parallel --sims "(1024, 2048, 4096, 8192, 16384)" --solver_args "{'p': (1.0, 2.0)}"

# Grid Search over Power Mean parameters
uv run benchmark --env frozenlake_slip --solver "('sp_uct', 'gsp_uct')" --seeds 500 --parallel --sims "(1024, 2048, 4096)" --solver_args "{'p': (1.0, 2.0), 'c': (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)}"

# Taxi
uv run benchmark --env taxi_rain --solver uct --seeds 50 --parallel --sims "(512, 1024, 2048, 4096, 8192, 16384)" --solver_args "{'c': (60.0)}"
```

### Arguments

- `--env`: Environment ID (e.g., `frozenlake`, `taxi`). Supports Grid Search `"(env1, env2)"`.
- `--solver`: Solver ID (e.g., `uct`, `sp_uct`). Supports Grid Search `"(s1, s2)"`.
- `--sims`: MCTS simulations per move. Supports Grid Search `"(100, 200)"`.
- `--solver_args`: Dictionary string for solver hypers. Values can be grids `"{'p': (1.0, 2.0)}"`.
- `--log`: Boolean flag. If present, saves CSVs, videos, and logs to `results/`.
- `--episodes`: Episodes per seed (default: 10).
- `--seeds`: Number of random seeds for statistical significance (default: 5).
- `--parallel`: Enable parallel episode execution.
- `--show_tree`: Print the colorful MCTS tree to terminal during replay.

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
