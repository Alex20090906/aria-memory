# The fast-weight memory inside the inference engine

The LLM server in this research is **Strata** (https://github.com/Niko1221/Strata), version 0.1.52, base commit
`a42cc3d`, MIT licence. The research files call it "the LLM server" and use `llm` in identifiers and paths.

Chapters 2 and 3 of the research used a memory **inside** the thinker model: a CUDA kernel added to Strata's
inference engine, run after a layer's residual write. It applies a sparse fast-weight matrix to every token, can
capture which units fired, and in "head" mode adds values just before the output layer for tokens after an anchor.
Writes were computed in Python and loaded by the engine when the memory file changed.

## The patch

[`third_party/strata/`](../third_party/strata/) holds the complete change against `a42cc3d` (10 files, 707 lines
added, 4 removed), squashed from 12 research commits made on 2026-11-04 and 2026-11-05, plus `fmem.cu` and
`fmem.hpp` for reading. It applies cleanly to the upstream commit and reproduces the research engine's last state
byte for byte (checked 2026-11-13). It is under Strata's MIT licence, with its notice in
`third_party/strata/LICENSE.Strata`. What each part does, the commit list and how the research scripts' environment
variables map to the engine's are in [`third_party/strata/README.md`](../third_party/strata/README.md).

Not in the patch, because it is Python: the write rule, anchors, sleep consolidation, the weakest-token confidence
threshold and the echo signature (`experiments/llm_fastmem/`).

## Without the modified engine

| Need | Use |
|---|---|
| Long-term memory, answers with sources | the final system (chapter 4): `Listener`, `NetIndex`, `read`; any LLM through `ask(messages, max_tokens)` |
| The index in network weights, but outside the LLM | `NetIndex` (numpy; 65,536 × 512 float32, 128 MB) |
| Recalling inside the model's own reasoning (inline recall, chapter 2.4) | needs the patched engine; the same index math could also run as a read head after layer 0 (engineering work, chapter 8.2) |
| The small-model experiments of chapters 2 and 3 | `experiments/memory/` uses Qwen2.5-1.5B through PyTorch hooks and needs no engine change |
