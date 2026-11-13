# 2. Fast-weight memory inside the thinker

**Claim.** A frozen LLM can learn new facts during use, without backpropagation and without harming its language,
if the memory is a sparse fast-weight layer: keys from the layer-0 word vectors (closest to the input, untouched by
context), values added just before the output layer, writes by an error-driven delta rule. Such a memory recalls
facts from natural questions and lets the model reason on the recalled words in the same pass. Its limits are
capacity under heavy load and "confident but wrong" answers once units saturate; chapter 4 shows the design that
replaced it for long-term storage.

Data: `syn-text` (English fact lists and an English "one year of life" script) unless noted.

## 2.1 Sparse fast weights versus LoRA (B2)

Qwen2.5-1.5B frozen; a fast-weight memory on the residual stream of layer 14: a fixed random expansion to 16,384
units with 32 active per token (pattern separation), then a learned matrix B added back. Each event: read the
sentence once, 5 update steps on its own language-model loss, B decays with τ = 40 events. 150 events, 3 seeds.
Recall = the most likely place among 10 (chance 0.10).

| Events since the fact | 0–20 | 20–50 | 50–100 | 100–150 |
|---|---|---|---|---|
| No memory | 0.18 | 0.11 | 0.09 | 0.07 |
| Sparse fast weights | 0.72 | 0.26 | 0.09 | 0.06 |
| LoRA rank 16, same updates | 0.27 | 0.10 | 0.08 | 0.08 |
| Familiarity AUC (true event vs a similar lure), slow trace + sensory adaptation | 1.00 | 1.00 | 0.99 | 0.87 |
| Same, without sensory adaptation | 0.48 | 0.53 | 0.49 | 0.47 |

General-text NLL without memory 2.01; with the sparse memory 1.98; with LoRA 8.38.

Source: `experiments/階段報告_02.md`, "實驗 B2"; raw `experiments/memory/results_inllm/`; code
`experiments/memory/inllm.py`.

Two findings carried through the rest of the research:

- **Pattern separation is what keeps online learning from destroying the model.** Each event touches few synapses.
- **Familiarity needs its own slow channel, computed after subtracting the recent average state** ("sensory
  adaptation"). Without it every sentence lands on the same units and the familiarity AUC is chance (0.48).

## 2.2 Write direction and key placement (H4, E4)

- Writing the value as the word-vector direction of the target reached recall 0.33; writing the true gradient
  direction `Jᵀ(u_y − E_p[u])` reached 0.97. A transpose was enough; the pseudo-inverse that was proposed was not
  needed. Source: `experiments/階段報告_04.md`, H4 table.
- With keys taken from a middle layer, a natural question ("We're late, where is Leo's passport?") recalled 0%: the
  context changes the deep representation. Keys built as a bilinear binding (owner × object) of **layer-0** word
  vectors recalled 92.5% on the small model and 100% of 40 facts inside the LLM server. Source: research map row E4
  (`experiments/map/研究地圖.md`) and `experiments/階段報告_05.md`.

This is the hippocampal-index idea in engineering form: index on the representation closest to the input, write the
content at the output.

## 2.3 Capacity at 200 facts: A1 and A2 (E5)

At 200 facts the errors were all first tokens overridden by the majority (behind the bookshelf, in the bookshelf).
Two changes:

- **A1, cross-shifted keys:** the same person with a different object moves the whole key, so frequent names stop
  sharing hot units.
- **A2, error-driven writes:** read what the memory already returns for the key, write only the difference.

| Key | Write | Clean anchor | Natural question |
|---|---|---|---|
| original | original | 55.5% | 69.5% |
| original | A2 | 91.0% | 91.5% |
| A1 | original | 74.5% | 93.0% |
| A1 | A2 | 95.0% | 94.5% |

Source: `experiments/階段報告_06.md`, table in section 二.2; raw `experiments/llm_fastmem/results/exp_c_n200_*.json`.
A later key change (answer tokens keyed by anchor plus step, not by the previous token, so one wrong token does not
derail the rest) gave 97% on natural questions at 200 facts (log entry 2026-11-05, "新鍵（不看前一字）"). Under
heavy load size alone does not fix it: making the memory 4 times larger (16,384 to 65,536 units) at 1,200 events
only halved the confident errors, 33 to 15 (chapter 3; row H5 of the research map).

## 2.4 Inline recall: no text hand-off (E3)

The thinker writes an anchor such as `[Leo|passport] ->` in its own reasoning; the engine adds the memory before the
output layer, the next tokens are the recalled place, and the model keeps reasoning ("it will be frosted, check it
for damp"). Recall 10/10, two-hop reasoning 8/8; in a 25-question simulated-world integration test, inline recall
25/25 and no memory 0/25. Source: `experiments/階段報告_06.md`, section 一 and 二.4;
`experiments/llm_fastmem/results/inline_recall_2026-11-04.md`; `experiments/integration/results3/`.

Familiarity gate: if the anchor units' familiarity θ is below 0.5 the memory is not injected and the model says it
does not remember. With the gate, colloquial questions about never-told things were answered honestly 40 of 40
(`experiments/階段報告_06.md`, section 一).

## 2.5 Confidence signals

| Signal | Result | Source |
|---|---|---|
| First-token probability ≥ 0.5 (H14) | confident errors 15 to 7, but right answers 26 to 19: it rejects true memories whose first word is unusual | log row H14; map E12 |
| Weakest token of the whole answer ≥ threshold (H14b) | AUC 0.82–0.96 on the small model (Kaggle runs); LLM server, 1,200 events, threshold 0.5: right 31, confident-wrong 0 (without: 34 and 14) | map rows H14b, E15, E18; `experiments/cloud/results_gputw/real_sleep_summary.txt` |
| Echo signature: does the anchor's unit set match the stored one (H15) | told vs never-told AUC 1.00 (small model); in the LLM server with no decay right answers 14 to 50, but answered never-told questions 1 to 7, mostly with high confidence | map rows H15, E18; `experiments/cloud/results_gputw/real_never_echo_summary.txt` |
| Familiarity decaying in step with content (H16) | confident errors 15 to 2, right answers 26 to 13 | map rows H16, E13 |

The weakest-token threshold generalised imperfectly: with another seed, threshold 0.5 still left 4 confident errors
(22 to 4); 0.9 reached 1 (map rows H21, E19). Long answers are rejected more at 0.9 (16% of short versus 69% of long
right answers on the small model; E26). Confidence is a usable knob at answer time, not a guarantee.

## 2.6 What did not work

| Idea | Result | Source |
|---|---|---|
| Error-correcting codes for stored answers (H17) | similar to analog values at light load, worse at high load; 3 of 750 collapsed at D = 4,096 | map rows H17, E15; `experiments/memory/results_meta/` |
| Unit-direction coherence as confidence (H18) | AUC 0.51–0.67 | map row H18 |
| Block addressing "8 times longer retention" (H19) | wrong derivation: the occupied fraction stays the same | map row H19 |
| Compiling fast weights into a dense slow matrix with null-space protection (H20) | 20 facts already raised general NLL 1.42 to 1.83 (512 protected directions); 100 facts gave 13.4 and answered "in the toolbox" to the capital of France | log entry 2026-11-05 "編譯回慢權重"; map E14 |
| Free attribute search guided by familiarity (H12) | recall 64% to 71%, but honesty on never-told questions 100% to 33%: familiarity alone creates false memories | log row H12 |

The compile result means that long-term memory has to stay sparse, keyed by anchors and gated; a dense matrix inside
the model is the wrong target.

## 2.7 Engineering inside the engine

The memory was added to the open-source inference engine behind the LLM server (MIT licence) as a CUDA kernel after the layer write: apply, capture and
anchor-locked head modes (`third_party/strata/`, [docs/ENGINE.md](../ENGINE.md)). Lessons that cost time:

- The engine's row positions differed from the prompt token count by 4 depending on cache state; anchors are now
  located by the arrow token, then relative to the last prompt token.
- The engine detected memory-file changes by whole-second timestamps, so an erase and a write in the same second were
  missed (H10); after the fix, updated facts were answered with the old value 1 of 18 times
  (log entry 2026-11-05, one-year test).
- Per-row scale vectors and dirty-row reload cut a 100-event run from 689 s to 475 s with identical answers (E16).
  The bottleneck was the thinker calls. All of these engine changes are in the published patch.
