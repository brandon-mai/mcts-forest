# AlphaZero JAX GPU Training Handoff Guide

This guide details instructions on how to run, monitor, and debug the optimized concurrent JAX training pipeline on a machine with a high-end GPU (RTX 5090).

## 1. Running Training

Run standard training commands. Example for `fourrooms`:
```bash
uv run train --env fourrooms --sims 128 --parallel_envs 32 --max_steps 50000 --batch_size 128 --buffer_size 50000 --learning_rate 5e-4 --save_freq 1000 --checkpoint_dir "results/gspuct_checkpoints"
```

To run training for the complex environment `2048`:
```bash
uv run train --env 2048 --sims 128 --parallel_envs 32 --max_steps 50000 --batch_size 128 --buffer_size 100000 --learning_rate 5e-4 --save_freq 1000 --checkpoint_dir "results/gspuct_2048_checkpoints"
```

## 2. Hardware Monitoring

To verify both CPU and GPU are being utilized effectively, monitor the system during the run:

### GPU Monitoring
Run the following command in a separate terminal:
```bash
nvidia-smi -l 1
```
*   **GPU Utilization:** Look at `% Volatile GPU-Util`. The asynchronous pipeline should keep this high (e.g. 70-95%) due to batch prefetching and background generation.
*   **VRAM Utilization:** Look at `Memory-Usage`. It should remain stable because the large replay buffer has been offloaded to host RAM (CPU side). You can scale `--parallel_envs` to larger values (e.g., 64, 128) if you want to saturate the 32GB VRAM.

### CPU Monitoring
*   Open **Task Manager** (Performance tab -> CPU) or run `htop` (if WSL).
*   Multiple cores should be active running Python threads (Actor thread doing self-play logic and coordination).

## 3. What to Look Out For & Debugging

*   **GPU Out-of-Memory (OOM):**
    *   If you see `Out of Memory` errors, decrease `--parallel_envs` (e.g. from 32 to 16) or reduce `--sims` (number of tree search simulations).
    *   To allow dynamic VRAM growth instead of JAX preallocating 75% upfront, set this environment variable:
        ```bash
        $env:XLA_PYTHON_CLIENT_PREALLOCATE="false"
        ```
*   **XLA Compilation Overhead:**
    *   The first iteration will experience a pause (up to 30-60 seconds) while JAX compiles the computational graphs. This is expected. Subsequent iterations will execute immediately.
*   **Checkpoints:**
    *   Checkpoints are saved periodically in `--checkpoint_dir`. Verify files ending in `.msgpack` are generated.

## 4. Asynchronous Pipeline Design

To maximize hardware utilization, the training loop runs on a multi-threaded asynchronous actor-learner pattern:

```
[CPU Actor Thread]  ──► [CpuReplayBuffer (RAM)] ──► [CPU Sampler]
        ▲                                                │ (device_put)
        │ (read params)                                  ▼
   [params_lock]    ◄─────────────────────────── [GPU Learner Thread]
```

### Key Components
1.  **Actor Thread (`actor_loop`):** Runs MCTS self-play continuously in the background on the CPU (using native vectorized JAX compiled calls). Pushes results to the host memory (`CpuReplayBuffer`).
2.  **Learner Loop (Main Thread):** Samples a list of mini-batches from host RAM using NumPy, stacks them, pushes the stacked block to GPU via `jax.device_put`, and applies JIT-compiled optimization steps (`train_steps_jit`).
3.  **Synchronization Locks:**
    *   `params_lock`: Protects reading/writing model weights. Actor reads under this lock; Learner updates under this lock.
    *   `buffer_lock`: Protects CPU-based Replay Buffer addition and sampling from concurrent access.

### Concurrency Tuning & Troubleshooting
*   **Starvation Tuning:** The Actor thread has a `time.sleep(0.01)` delay. If CPU utilization is low and the GPU is starved (waiting for samples), reduce this delay to `0.001` or remove it. If CPU usage is 100% and training lags, increase this sleep delay.
*   **Lock Deadlocks:** Ensure any manual access to `params` or `buffer` inside code changes holds locks briefly and releases them immediately. Never call blocking or slow JAX compilations inside a lock block.
