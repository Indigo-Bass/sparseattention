import torch
import time
import matplotlib.pyplot as plt
from attention import dense_attention, create_sliding_window_mask, create_bigbird_mask

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pre-filled hardware string for the requirement
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    hardware_info = f"ASUS TUF Gaming F16 (Intel Core i7) - {gpu_name}"
    
    print(f"=== Item 1.5: Benchmarking on {device.type.upper()} ===")
    print(f"Hardware: {hardware_info}\n")

    seq_lengths = [512, 1024, 2048, 4096, 8192]
    B, H, d_k = 2, 4, 16
    window_size, num_global, num_random = 128, 64, 32  # Realistic sparsity parameters

    results = {
        "Dense": {"time": [], "memory": []},
        "Sliding Window": {"time": [], "memory": []},
        "BigBird": {"time": [], "memory": []}
    }

    # Disable gradients for benchmarking forward pass only
    with torch.no_grad():
        for S in seq_lengths:
            print(f"Benchmarking Sequence Length: {S}...")
            Q = torch.randn(B, H, S, d_k, device=device)
            K = torch.randn(B, H, S, d_k, device=device)
            V = torch.randn(B, H, S, d_k, device=device)

            # Generate masks on device
            sw_mask = create_sliding_window_mask(S, window_size, causal=True).to(device)
            bb_mask = create_bigbird_mask(S, window_size, num_global, num_random, causal=True).to(device)

            configs = [
                ("Dense", None),
                ("Sliding Window", sw_mask),
                ("BigBird", bb_mask)
            ]

            for name, mask in configs:
                # Warmup
                for _ in range(3):
                    dense_attention(Q, K, V, mask=mask)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.perf_counter()
                
                # Run timed iterations
                iters = 10
                for _ in range(iters):
                    dense_attention(Q, K, V, mask=mask)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    mem_mb = 0.0 # Memory tracking is CUDA-specific in this setup
                
                end_time = time.perf_counter()
                
                avg_time_ms = ((end_time - start_time) / iters) * 1000
                results[name]["time"].append(avg_time_ms)
                results[name]["memory"].append(mem_mb)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Sparse Attention Benchmark\nHardware: {hardware_info}")

    for name in results.keys():
        ax1.plot(seq_lengths, results[name]["time"], marker='o', label=name)
        if device.type == "cuda":
            ax2.plot(seq_lengths, results[name]["memory"], marker='s', label=name)

    ax1.set_title("Forward Pass Wall-Clock Time")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Time (ms)")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Peak Memory Allocated")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Memory (MB)" if device.type == "cuda" else "Memory Tracking Unavailable on CPU")
    if device.type == "cuda":
        ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\n[PASS] Benchmark complete! Saved plot to 'benchmark_results.png'")

if __name__ == "__main__":
    run_benchmark()