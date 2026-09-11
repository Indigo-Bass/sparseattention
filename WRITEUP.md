# Sparse Attention from Scratch: Implementation and Analysis

## 1. Overview and Implementations
*(Write 2-3 paragraphs here. Explain that you implemented manual scaled dot-product attention as a baseline, followed by two sparse masks: Sliding Window and a BigBird-style block-sparse pattern. Mention how you generated the boolean masks and integrated them seamlessly.)*

## 2. Correctness and NaN Edge Case Handling (Item 1.4)
*(Write 1-2 paragraphs. Explicitly state that when causal masking combines with aggressive sparsity, certain tokens can be entirely masked out. Explain that this results in rows of `-inf` before the softmax, which evaluate to `NaN` (0/0). Detail your fix: using `torch.nan_to_num` to zero out the attention weights for fully masked tokens so they contribute nothing, rather than corrupting the matrix.)*

## 3. Performance Benchmark (Item 1.5)
*(Insert your plot here using markdown: `![Benchmark Results](benchmark_results.png)`)*
*(Write 2 paragraphs analyzing the time and memory complexity. Discuss how standard dense attention scales quadratically $O(N^2)$, whereas your sparse patterns cap the context, theoretically leaning toward linear scaling. Note any practical overhead from mask generation.)*

## 4. Information Loss and Global Tokens (Item 1.7)

### Which pattern loses what information?
*(Discuss Sliding Window: It loses long-range dependencies. A token at position 500 has zero context about what happened at position 10. Discuss BigBird: It recovers some long-range context via global/random tokens but loses the dense, rich middle-distance context.)*

### Why do global tokens matter disproportionately?
*(Explain that global tokens act as information hubs or routing bottlenecks. In a language model, the first few tokens often set the topic, system prompt, or syntactic structure. Allowing all tokens to attend to these global tokens grounds the sequence, preventing the model from losing the overall plot even if local windows are small.)*

### Was Sparse Attention Worse?
*(Be honest here as the prompt requests. Acknowledge that yes, mathematically, sparse attention is an approximation. By forcefully masking out values, you are actively destroying context that a dense model would use to fine-tune its representations. It trades representational capacity for computational efficiency.)*