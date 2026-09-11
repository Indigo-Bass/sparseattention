import math
import torch
import torch.nn.functional as F

def dense_attention(Q, K, V, mask=None):
    """
    Computes manual scaled dot-product attention:
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + mask) @ V

    Args:
        Q: Tensor of shape (..., S_q, d_k)
        K: Tensor of shape (..., S_k, d_k)
        V: Tensor of shape (..., S_k, d_v)
        mask: Optional tensor broadcastable to (..., S_q, S_k).
              Can be boolean (True = mask out / ignore) or float (-inf for masked).
    
    Returns:
        output: Tensor of shape (..., S_q, d_v)
        attn_weights: Tensor of shape (..., S_q, S_k)
    """
    d_k = Q.size(-1)
    
    # 1. Scaled dot-product: (..., S_q, d_k) @ (..., d_k, S_k) -> (..., S_q, S_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            # Mask out positions where mask == True with -inf
            scores = scores.masked_fill(mask, float('-inf'))
        else:
            scores = scores + mask

    # 3. Softmax along the key sequence dimension
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 4. Weighted sum over values: (..., S_q, S_k) @ (..., S_k, d_v) -> (..., S_q, d_v)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

def create_sliding_window_mask(seq_len, window_size, causal=False):
    """
    Creates a boolean mask for sliding window attention.
    Returns: Tensor of shape (seq_len, seq_len) where True means MASKED OUT (ignored).
    """
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    
    # Keep tokens within window distance
    keep = torch.abs(i - j) <= window_size
    
    if causal:
        # If causal, a token can only attend to previous tokens
        keep = keep & (j <= i)
        
    return ~keep

def create_bigbird_mask(seq_len, window_size, num_global, num_random, causal=False):
    """
    Creates a boolean mask for BigBird-style sparse attention.
    (Local window + Global tokens + Random tokens)
    """
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    
    # 1. Local connections
    keep_local = torch.abs(i - j) <= window_size
    
    # 2. Global connections (e.g., first few tokens are highly connected)
    keep_global = (i < num_global) | (j < num_global)
    
    # 3. Random connections (each row gets a few random tokens it can attend to)
    rand_scores = torch.rand(seq_len, seq_len)
    _, random_indices = torch.topk(rand_scores, num_random, dim=-1)
    keep_random = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    keep_random.scatter_(1, random_indices, True)
    
    # Combine all allowed connections
    keep = keep_local | keep_global | keep_random
    
    if causal:
        keep = keep & (j <= i)
        
    return ~keep



if __name__ == "__main__":
    # Quick sanity check on random tensors
    torch.manual_seed(42)
    B, H, S, d_k = 2, 4, 8, 16
    
    Q = torch.randn(B, H, S, d_k)
    K = torch.randn(B, H, S, d_k)
    V = torch.randn(B, H, S, d_k)
    
    out, weights = dense_attention(Q, K, V)
    
    print("--- Sanity Check ---")
    print(f"Output shape  : {out.shape} (Expected: ({B}, {H}, {S}, {d_k}))")
    print(f"Weights shape : {weights.shape} (Expected: ({B}, {H}, {S}, {S}))")
    print(f"Row sums check: {weights.sum(dim=-1)[0, 0].tolist()[:4]} (Expected: ~1.0)")
    
    # Verify matches PyTorch reference when no mask is applied
    ref_out = F.scaled_dot_product_attention(Q, K, V)
    diff = torch.max(torch.abs(out - ref_out)).item()
    print(f"Max absolute difference vs F.scaled_dot_product_attention: {diff:.2e}")
    assert diff < 1e-5, "Mismatch against PyTorch reference implementation!"
    print("Item 1.1 test passed successfully.")