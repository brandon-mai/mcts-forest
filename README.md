# Monte Carlo Forest 🌲

### Neural Solvers (AlphaZero, MuZero, Stochastic MuZero)
Unified under the `GSP` prefix, these solvers use Numba-accelerated search cores and PyTorch neural networks.

## Kaggle & Cloud Execution
The framework is optimized for both local CPU development and Kaggle GPU training.

### Local Development (Smoke Test)
Run a quick iteration locally to verify the triple-variant logic:
```bash
uv run train --algo gsp_alphazero --iterations 1 --games_per_it 1 --sims 10 --wandb
```

### Kaggle GPU Deployment
1.  **Accelerator**: Select `GPU P100` (recommended).
2.  **Persistence**: Enable **Files Persistence** in Settings.
3.  **Environment Setup**:
    ```python
    !pip install -q uv
    !git clone <your-repo-url>
    %cd mcts-forest
    !uv pip install --system -e .
    ```
4.  **Triple-Variant Training**: By default, the script trains three concurrent models:
    - **GSP**: Trained on GSP-solver data, evaluated with GSP.
    - **Baseline**: Trained on Tree-search data, evaluated with Tree-search.
    - **Cross**: Trained on Tree-search data, evaluated with GSP.
    ```python
    # Use python -m to avoid 'uv run' creating a large venv in the output folder
    !python -m mcts_forest.scripts.train_loop --algo gsp_alphazero --env taxi --use_amp --wandb --wandb_key "YOUR_KEY_HERE"
    ```

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
