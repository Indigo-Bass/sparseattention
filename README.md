# Sparse Attention from Scratch

This repository contains a manual implementation of scaled dot-product attention and two sparse variants (Sliding Window and BigBird-style block-sparse), built entirely from scratch in PyTorch without relying on `F.scaled_dot_product_attention`.

## Repository Structure

* `attention.py`: The core mathematical implementation of dense attention, NaN edge-case handling, and boolean mask generation for sparsity patterns.
* `harness.py`: The strict correctness harness. It asserts that all sparse variants match PyTorch's optimized backend on unmasked positions within a `1e-5` tolerance.
* `benchmark.py`: Profiling script that measures forward-pass wall-clock time and peak GPU memory allocation across sequence lengths 512 to 8192.
* `train.py`: A 2-layer character-level GPT training loop evaluating the quality (validation loss) of the dense vs. sparse patterns.
* `WRITEUP.md`: A comprehensive analysis of performance tradeoffs, why global tokens matter, and the empirical realities of naive masking overhead.
* `benchmark_results.png` & `training_logs.txt`: Artifacts proving execution on a T4 GPU.

## How to Run

### 1. Install Dependencies

```bash
pip install torch matplotlib
```

### 2. Run the Correctness Harness

Verify numerical correctness and fully-masked row (NaN) handling:

```bash
python harness.py
```

### 3. Run the Benchmark

Generate the performance metrics and export the plot:

```bash
python benchmark.py
```

### 4. Run the Training Evaluation

To run the 2-layer GPT training loop, first download the TinyShakespeare dataset, then execute the script (a CUDA GPU is highly recommended to complete this in ~10 minutes):

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python train.py
```