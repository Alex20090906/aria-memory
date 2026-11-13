# Aria memory: research notes

This folder is the paper-style write-up of the "digital life" memory research (2026-11-03 to 2026-11-09). Each
chapter stands alone. Every number names its source: a raw file in [`experiments/`](../../experiments/), or an entry
of the research log [`experiments/研究總紀錄.md`](../../experiments/研究總紀錄.md) (Traditional Chinese, dated
entries). Experiment, hypothesis and problem ids (E49, H35, P8 ...) are the ids of the research map
[`experiments/map/研究地圖.md`](../../experiments/map/研究地圖.md), which lists every node with its result in one line.

## Data provenance

**No number in this repository comes from a real person's voice, life or home.** All inputs are synthetic:

| Tag | What it is | Where |
|---|---|---|
| `syn-text` | Text written by an LLM (Gemini or Claude) or by templates: English fact lists, an English "one year of life" script, a simulated world | `experiments/colloquial/`, `experiments/memory/`, `experiments/model1/` |
| `syn-speech` | Two 21-day Taiwanese-Mandarin household scripts (Gemini wrote them), spoken by Windows OneCore text-to-speech voices (owner, family, visitor, TV), mixed with synthetic room noise, then transcribed by faster-whisper and checked by a speaker-verification model | scripts, timelines and transcripts in `tests/fixtures/set_old/`, `tests/fixtures/set2/`; generator in `experiments/realistic/` |
| `syn-day` | A 21-day always-on timeline built from the `syn-speech` transcripts plus synthetic TV, family, visitor and appliance sounds (also run through the same speech recognition) | `tests/fixtures/alwayson/`, `experiments/alwayson/` |

The audio files are not in the repository (see [`docs/REVIEW.md`](../REVIEW.md) for how to regenerate them
with any TTS). The transcripts that every number was computed from are included.

The thinker (the LLM that listens, reads and judges) was an **LLM server** running Qwen3.8-Flash-Next (IQ3_S) unless a
chapter says otherwise. Small-model experiments used Qwen2.5-1.5B-Instruct.

## Chapters

1. [Forced generation and the rest state](01-forced-generation-and-rest-state.md): a language model that must keep
   generating either fixates or invents inputs; an always-on system needs a legal "do nothing" state.
2. [Fast-weight memory inside the thinker](02-fast-weight-memory.md): sparse fast weights with keys at layer 0 and
   values at the output; capacity fixes A1/A2; familiarity and confidence signals; what failed.
3. [Sleep consolidation](03-sleep-consolidation.md): every forgetting and consolidation run in one table.
4. [The index and the record log](04-index-and-record-log.md): the final design (E49). The network finds, an
   append-only log keeps the words, the thinker reads.
5. [The gate and the energy pool](05-gate-and-energy-pool.md): 21 simulated days always on, vegetative-state
   metrics, positive controls.
6. [Listening under messy input](06-listening.md): where the remaining losses are (hearing, not storing).
7. [Negative results](07-negative-results.md): hypotheses that did not hold, with numbers.
8. [Limitations and open problems](08-limitations.md), including the relation to public long-term-memory benchmarks
   (not run).

## The final system in one paragraph (E49)

Speech is transcribed and checked for the owner's voice. A listening step (the thinker, one short call per owner
sentence) writes facts as records `thing | aspect | value` with the sentence it heard, holding each new fact for two
more sentences so a correction can replace it. Records are never edited. A sparse associative memory (65,536 units,
512-dimensional codes, delta-rule writes) maps the words and sounds of each record to the record's position; a
question reads out the five best records, and the thinker answers only if one of them is clearly about the same thing
and aspect, otherwise it says it does not know. At night the thinker re-reads the day and writes missed facts. An
energy pool and four fuses decide when the system may think on its own while nobody talks to it.

| Result (E49, `syn-speech` set_old, 21 simulated days) | Value | Source |
|---|---|---|
| Told questions answered right / wrong, strict scoring | 30 / 6 of 51 | `tests/fixtures/alwayson/e49_answers.json`, rescored by `aria_memory.eval.score` (`examples/quickstart.py` prints it) |
| Never-told questions held ("I don't know") | 20 of 21 | same |
| Right record in the index's top 5 | 35/38 (set_old), 34/38 (set2) | `experiments/p8/eval_h41_ortho.txt`; reproduced by `tests/test_index.py::TestRealVocab` with the model vocabulary |
| Reading time per answer, median | 1.54 s | median of `mem_calls.read_s` (94 reads) in `experiments/alwayson/results/run_Dd_21d_s0_e49_final.json`; log entry 2026-11-06 E49 |
| Vegetative hours | 2 of 504 | row `Dd_21d_s0_e49_final` of `experiments/alwayson/results/summary.md` (made by `experiments/alwayson/analyze.py`); definition in chapter 5 |

The answer rate is low because hearing is the bottleneck: on the same scripts, 16% of owner questions never reached
the system in early runs because voice ID missed the owner, and facts said in passing are often mis-transcribed
(chapter 6). Storage itself was close to perfect in the long-term runs: 56/60 told facts right, 0 confidently wrong,
after 2,400 events (E39, chapter 4).

## Credits

The research goal and questions are the researcher's. Experiments, code and write-up: Claude (Anthropic). Several
methods came from design rounds with Gemini (Google); each was tested, and chapter 7 lists the ones that failed. The
researcher's own conjectures and the discussion records are not part of this release.
