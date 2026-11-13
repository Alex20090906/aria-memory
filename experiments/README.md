# experiments/: the raw research record

Code, logs, result files and reports of the research, 2026-11-03 to 2026-11-09, as they were, with the changes
listed in [docs/REVIEW.md](../docs/REVIEW.md#2-what-was-changed-in-copied-research-files). Most prose is in
Traditional Chinese (one early report, `阶段报告_01.md`, is in Simplified Chinese). The English write-up is
[docs/research/](../docs/research/).

Most scripts here expect the research machine: an LLM server with Qwen3.8-Flash-Next on a GPU, sometimes the
modified engine, fixed folders. They are here so that every number can be traced, not as a runnable package. The
scoring scripts named below run on a CPU from stored outputs.

## Start here

| File | What |
|---|---|
| `研究總紀錄.md` | the research log: hypotheses (H), test-realism audit (T) and a dated entry for every experiment (the researcher's conjectures U1–U18 are not included; U numbers remain as labels) |
| `map/研究地圖.md` | the research map: one line per goal, hypothesis, experiment, component, problem and platform, with its result and links (generated from `map/research_map.json` by `map/map_tool.py`) |
| `阶段报告_01.md` … `階段報告_06.md`, `問題清單與擴大計畫.md` | stage reports and the problem list written during the research |

## Folders

| Folder | Experiments | Chapter |
|---|---|---|
| `habituation/` | E1 forced continuous generation (`results_a/`) | 1 |
| `model1/` | the reflex model and its simulated worlds (C1–C4, E2, E9); results per world in `results*/` | 1 |
| `memory/` | small-model memories: engram v0, in-LLM fast weights vs LoRA (B2), anchor keys, compile to slow weights (H20), confidence and error-correcting codes (`results_meta/`, H14b, H17) | 2 |
| `llm_fastmem/` | the memory inside the LLM server: anchor tools, sleep consolidation (`sleep.py`), colloquial and one-year tests, the Chinese listening pipeline (`zh_v3.py`) and its end-to-end results E17–E43 (`results/`) | 2, 3, 4, 6 |
| `cloud/` | long-load forgetting runs on rented GPUs (E18–E39): `results_gputw/`, `results_modal/`, rescored in `rescore_output.txt` | 3 |
| `integration/` | three-model integration in the simulated world (E3) | 2 |
| `colloquial/` | Gemini-written colloquial questions and the English one-year script | 2 |
| `realistic/` | the synthetic-speech test level: script generator, TTS rendering (audio not included), speech recognition, voice ID, transcripts; `set2/` is the second script | 4, 6 |
| `sensors/` | speech recognition, owner voice, camera captions, alias table; unit tests | 6 |
| `p8/` | offline evaluations of finding and reading (H25–H42): label search, retrieve-then-read, the network index, scale tests, night re-listening, supersede marks; outputs in `eval_*.txt` | 4, 6 |
| `alwayson/` | the 21-day always-on simulator (E20–E49): `sim.py`, timeline builder, background sound bank, `results/`, `summary.md` from `analyze.py`, strict rescoring and the latency table | 1, 4, 5 |
| `recording/` | the reading script prepared for real recordings (text only; no recording was made) | 8 |

## CPU-only checks from stored outputs

```bash
python experiments/alwayson/rescore_strict.py e44_read e45_read_idle e46_index e47_nogate e48_full e49_final
python experiments/alwayson/latency_table.py
python experiments/alwayson/analyze.py              # rewrites results/summary.md and summary.json
python experiments/cloud/rescore_gputw.py experiments/cloud/results_gputw/*.json experiments/cloud/results_modal/*.json
```

## Not included

By the researcher's decision, the researcher's conjectures U1–U18 (including the theory behind the research), the theory-integration and overall-design notes, and the records of the design rounds with Gemini are not published. Where a method came from a Gemini suggestion, the log keeps one line saying so. Scripts that called Gemini (`colloquial/gen_*.py`, `model1/gen_m1_pools.py`, `realistic/gen_zh_scripts.py`, `realistic/judge.py`) imported a small client from the removed folder and need a replacement client to run.
