# Fast-weight memory inside the LLM server engine (design)

Goal: memory model v1 (sparse fast weights + delta rule) runs inside the thinker itself, so memories do
not pass through text. Validated first on Qwen2.5-1.5B (`experiments/memory/compile_ctx.py`,
`--target logit`): forward passes only, no backpropagation.

## Split of work

| Part | Where | What |
|---|---|---|
| Apply | engine (CUDA) | after layer l's write, for each token: x = mean of the hc residual streams, LayerNorm, adapt `(x - mu_l) / sd_l`, scores = x · P_l (D units), top-k units, R += sum_j B_l[unit_j] / sqrt(k) on every hc stream |
| Capture | engine | when a capture file is set, write per token: position, token id, and the k unit ids per memory layer |
| Learn (compile) | Python | target_t = gain * normalize(U[y_{t+1}] - mean(U)); delta rule B_l[units] += eta / sqrt(k) * (target / n_layers - current_out); decay B with tau_w |
| Reload | engine | before each request, if the memory file changed, copy B (and mu, sd) into the same device buffers (pointers stay fixed, so captured CUDA graphs stay valid) |

## Hook point

`llm::kernels::cvec_apply` is already called after the layer write in every path:
decode (`core/layer.cpp: block_layer_post`), speculative verify (`core/verify.cpp`) and prefill
(`prefill/prefill.cpp`). The memory adds a second kernel right after it.

- `Cvec::covers(l)` must also be true for memory layers, so every path calls `cvec_apply` there.
- Without a control vector, an all-zero vector is uploaded, so `cvec_kernel` only does the pending
  residual write.
- The fused prefill path (`LLM_CVEC_FUSE`) is turned off for memory layers, so `cvec_apply` runs
  there unfused.

## Kernels

1. `fmem_scores`: grid (D / 256, ceil(T / 4)), each block holds 4 tokens' adapted x in shared memory;
   P is stored n x D (transposed) so the reads are coalesced. Output: scores [T x D] in a scratch buffer,
   processed in slices of 512 tokens.
2. `fmem_topk_add`: one block per token: k rounds of block argmax over D, then adds the k rows of B
   (D x n, fp16) to every hc stream; writes the unit ids to the capture buffer when capture is on.

Memory per layer at D = 16384, n_embd = 2048: P 64 MB + B 64 MB (fp16) + scratch 32 MB.

## Files

- `memory.bin`: header (magic, version, n_layers_mem, layer ids, D, k, n_embd), then per layer
  mu, sd (fp32, n), P (fp16, n x D, fixed), B (fp16, D x n).
- capture file: records (int32 pos, int32 token, int32 units[n_layers_mem * k]).

## Safety

- The new engine goes into its own folder; the installed LLM server is not replaced.
- With no memory file, the engine is bit-identical to the original: no layer is covered, no kernel runs.
