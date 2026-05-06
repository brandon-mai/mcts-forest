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
    !python -m mcts_forest.scripts.train_loop --algo gsp_alphazero --env taxi --use_amp --wandb --wandb_key "YOUR_KEY_HERE" --iterations 5 --games_per_it 50 --sims 400
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
uv run benchmark --env sysadmin_n10 --solver gbop --sims "(512, 1024, 2048, 4096, 8192, 16384)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 50}" --table

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
