import torch
import torch.nn.functional as F
from attention import dense_attention, create_sliding_window_mask, create_bigbird_mask

def test_correctness():
    torch.manual_seed(42)
    B, H, S, d_k = 2, 4, 16, 16
    Q, K, V = torch.randn(B, H, S, d_k), torch.randn(B, H, S, d_k), torch.randn(B, H, S, d_k)
    
    print("=== Item 1.3: Correctness Harness ===")

    # 1.1 Dense Check
    out_dense, _ = dense_attention(Q, K, V)
    ref_dense = F.scaled_dot_product_attention(Q, K, V)
    diff_dense = torch.max(torch.abs(out_dense - ref_dense)).item()
    assert diff_dense < 1e-5, "Dense mismatch!"
    print(f"[PASS] 1.1 Dense Attention matches PyTorch reference (Diff: {diff_dense:.2e})")

    # 1.2 & 1.3 Sparse Pattern Checks
    window_size, num_global, num_random = 2, 2, 1
    sw_mask = create_sliding_window_mask(S, window_size, causal=True)
    bb_mask = create_bigbird_mask(S, window_size, num_global, num_random, causal=True)

    for name, mask in [("Sliding Window", sw_mask), ("BigBird", bb_mask)]:
        out_sparse, _ = dense_attention(Q, K, V, mask=mask)
        ref_sparse = F.scaled_dot_product_attention(Q, K, V, attn_mask=~mask)
        
        # Temporarily handle NaNs for the comparison
        out_sparse = torch.nan_to_num(out_sparse, nan=0.0)
        ref_sparse = torch.nan_to_num(ref_sparse, nan=0.0)
        
        diff = torch.max(torch.abs(out_sparse - ref_sparse)).item()
        assert diff < 1e-5, f"{name} mismatch! Diff: {diff}"
        print(f"[PASS] 1.2 & 1.3 {name} matches PyTorch reference (Diff: {diff:.2e})")

def test_nan_handling():
    print("\n=== Item 1.4: NaN Edge Case Handling ===")
    B, H, S, d_k = 1, 1, 4, 16
    Q, K, V = torch.randn(B, H, S, d_k), torch.randn(B, H, S, d_k), torch.randn(B, H, S, d_k)
    
    # Construct a mask where the last row is entirely True (fully masked)
    mask = torch.zeros(S, S, dtype=torch.bool)
    mask[-1, :] = True 
    
    out, weights = dense_attention(Q, K, V, mask=mask)
    
    assert not torch.isnan(out).any(), "NaNs detected in output!"
    assert not torch.isnan(weights).any(), "NaNs detected in weights!"
    assert (out[0, 0, -1, :] == 0).all(), "Fully masked row should output zeros."
    
    print("[PASS] 1.4 Fully masked rows handled gracefully (NaNs converted to 0.0)")

if __name__ == "__main__":
    test_correctness()
    test_nan_handling()