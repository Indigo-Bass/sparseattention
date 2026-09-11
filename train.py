import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os

from attention import dense_attention, create_sliding_window_mask, create_bigbird_mask

# --- Hyperparameters ---
batch_size = 32
block_size = 128
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 64
n_head = 4
n_layer = 2

print(f"Running on device: {device.upper()}")

# --- Load Dataset ---
if not os.path.exists('input.txt'):
    print("Please download input.txt (TinyShakespeare) first.")
    exit()

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size, pattern, window_size=32, num_global=4, num_random=4):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.pattern = pattern
        self.window_size, self.num_global, self.num_random = window_size, num_global, num_random

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        v = self.value(x) 

        if self.pattern == 'dense':
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        elif self.pattern == 'sliding_window':
            mask = create_sliding_window_mask(T, self.window_size, causal=True).to(x.device)
        elif self.pattern == 'bigbird':
            mask = create_bigbird_mask(T, self.window_size, self.num_global, self.num_random, causal=True).to(x.device)
        
        out, _ = dense_attention(q, k, v, mask=mask)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, pattern):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, pattern) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, pattern):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, pattern)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
        )
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CharGPT(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, pattern) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        loss = None if targets is None else F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

def train_pattern(pattern):
    print(f"\n--- Training {pattern.upper()} ---")
    model = CharGPT(pattern).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_t = time.time()
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    train_time = time.time() - start_t
    final_loss = estimate_loss(model)['val']
    print(f"[{pattern.upper()}] Final Val Loss: {final_loss:.4f} | Time: {train_time:.1f}s")
    return final_loss

if __name__ == "__main__":
    results = {}
    for p in ['dense', 'sliding_window', 'bigbird']:
        results[p] = train_pattern(p)

    print("\n=== FINAL VALIDATION LOSS ===")
    for p, loss in results.items():
        print(f"{p}: {loss:.4f}")