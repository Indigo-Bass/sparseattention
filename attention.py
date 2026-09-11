import math
import torch
import torch.nn.functional as F

def dense_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float('-inf'))
        else:
            scores = scores + mask

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

def create_sliding_window_mask(seq_len, window_size, causal=False):
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    
    keep = torch.abs(i - j) <= window_size
    if causal:
        keep = keep & (j <= i)
        
    return ~keep

def create_bigbird_mask(seq_len, window_size, num_global, num_random, causal=False):
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    
    keep_local = torch.abs(i - j) <= window_size
    keep_global = (i < num_global) | (j < num_global)
    
    rand_scores = torch.rand(seq_len, seq_len)
    _, random_indices = torch.topk(rand_scores, num_random, dim=-1)
    keep_random = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    keep_random.scatter_(1, random_indices, True)
    
    keep = keep_local | keep_global | keep_random
    if causal:
        keep = keep & (j <= i)
        
    return ~keep