# aria-memory

> Part of the research *An Always-On Personal AI Voice Assistant: System Architecture and a Novel Memory System, with
> a Study of Time Perception in Large Language Models*. This repository is the memory system. The main entry point
> is `always-on-assistant`.
>
> 繁體中文說明：[README.zh-TW.md](README.zh-TW.md)

**A memory for an always-on voice assistant: an append-only log of what the owner said, a sparse associative index
in network weights that finds the right records from words and sounds, and a strict reader that answers only from
them, plus a gate that lets the assistant think on its own without falling into a loop.**

Research code. Pure Python and numpy, runs on a CPU. The LLM that listens and reads is yours: any function
`async ask(messages, max_tokens) -> str`.

## Quick try (no GPU, no model download, about 10 seconds)

```bash
git clone <this repository> && cd aria-memory
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[test]"
python examples/quickstart.py
pytest -q
```

`examples/quickstart.py` builds an index over a few spoken records and queries it, re-scores the final research
system's logged answers, runs the index over two 21-day test scripts, and shows what a wake-word-only microphone
would still hear. It uses `CharVocab`, a vocabulary that needs no model. Expected output (abridged):

```
2. Strict re-scoring of E49's logged answers (tests/fixtures/alwayson/e49_answers.json)
   told questions: right 30, wrong 6, of 51
   never-told questions held (said 'I don't know'): 20 of 21
3. Index recall on the 21-day scripts with CharVocab (no model) (a right record in the top 5?)
   set_old: 34/38   (research, model vocabulary: 35/38)
   set2: 34/38   (research, model vocabulary: 34/38)
```

`pytest -q`: 48 tests pass and 4 are skipped without the model vocabulary; with it (see
[docs/VOCAB.md](docs/VOCAB.md)) all 52 pass, including exact parity with the research numbers.

## Results

All inputs are synthetic: LLM-written scripts, Windows text-to-speech voices, generated room noise, real speech
recognition. No real person's voice or life was used. Details per number: [docs/research/](docs/research/).

| Result | Value | Data | Reproduce or read |
|---|---|---|---|
| Told questions right / wrong, strict scoring (final system E49, 21 simulated days) | 30 / 6 of 51 | synthetic speech | `python examples/quickstart.py`; `tests/test_replay.py::test_strict_rescoring_reproduces_e49` |
| Never-told questions answered "I don't know" | 20 of 21 | synthetic speech | same |
| Right record in the index's top 5, model vocabulary | 35/38, 34/38 | synthetic speech | `tests/test_index.py::TestRealVocab` (needs the vocabulary); log `experiments/p8/eval_h41_ortho.txt` |
| Same, `CharVocab` (no model) | 34/38, 34/38 | synthetic speech | `python examples/quickstart.py` |
| Same, with 10,000 same-name hard distractors: network index vs a text-embedding model | 18 vs 15, 29 vs 14 | synthetic speech | log `experiments/p8/eval_index_scale_dense.txt` |
| Long-term storage, 60 facts among 2,400 events: right / honest forget / confidently wrong | 56 / 4 / 0 (oldest quarter 15/15) | synthetic text | log `experiments/cloud/results_modal/sleep_never_n2340_u15_never_echo_2400.json`; rescored in `experiments/cloud/rescore_output.txt` |
| Same load with facts stored in fast weights and sleep consolidation | 15 / 18 / 27 | synthetic text | log `experiments/cloud/results_gputw/load2400__sleep_sleep_n2340_load2400.json` |
| Reading time per answer, median / 90th percentile | 1.54 s / 1.90 s | thinker: Qwen3.8-Flash-Next on an RTX 3080 | log `experiments/alwayson/latency_table.txt` (`latency_table.py`) |
| Vegetative hours in 21 days: gated always-on vs no brakes | 2 of 504 vs 72 of 72 | simulated days | log `experiments/alwayson/results/summary.md` |
| Forced continuous generation: questions answered with a repetition penalty | 2 of 9 (NLL 0.40 to 2.80) | synthetic text | log `experiments/habituation/results_a/`, report `experiments/阶段报告_01.md` |
| Wake word only: told questions still heard; with an 8 s follow window | 24 of 51; 44 of 51 | synthetic speech | `python examples/quickstart.py` |

Rows marked "log" come from runs that needed the thinker model (a 78 GB model on a GPU) and are not re-run here; the
raw output is in the repository and the scoring scripts run on a CPU.

## How it works

```
speech -> transcription -> owner's voice? -> listening (LLM, one short call per sentence)
                                                  |  facts: thing | aspect | value  (+ the sentence heard)
                                                  v
                      append-only record log  +  index in network weights (words and sounds -> record)
                                                  |
question -> index proposes 5 records -> reader (LLM): answer only if one is about the same thing and aspect
night: re-listen to the day, write missed facts; optionally mark records that a newer one replaces
idle:  a gate with an energy pool and four fuses decides when the assistant may think on its own
```

| Module (`src/aria_memory/`) | Job |
|---|---|
| `listen.py`, `rules.py` | turn one owner sentence into records; staged commit (a correction within 2 sentences replaces); relative dates by rule; drop questions written as facts |
| `index.py`, `vocab.py` | the index: sparse keys over 65,536 units (meaning units from a model's layer-0 word vectors, or `CharVocab`; pinyin sound units), 512-d record codes, delta-rule writes, save/load, real deletion |
| `read.py`, `records.py` | how records are shown to the reader; retrieve-then-read with a second read for exact sound-alikes |
| `sleep.py` | night re-listening; strict marking of replaced records |
| `gate.py`, `reflex_gru.py` | the always-on gate (energy pool, fuses, idle work, proactive speech cap); the research reflex model in numpy |
| `eval.py`, `replay.py` | the 21-day scripts, strict scoring, wake-word coverage, cached replays, acceptance reports |

Prompts are in Traditional Chinese (`prompts.py`): the research targeted Taiwanese Mandarin households. The
assistant's name inside the prompts and the default wake word is `NAME = "Aria"`.

### Use it with your LLM

```python
from aria_memory import CharVocab, Listener, NetIndex, read

async def ask(messages, max_tokens):     # any chat model: return the reply text (no thinking)
    ...

listener, index, records = Listener(), NetIndex(CharVocab()), {}
r = await listener.hear(ask, "護照我改放黑色後背包夾層了", day=3, src="owner")
for rec in r.records + listener.flush():          # facts wait 2 sentences for a correction; flush at day end
    rec["id"] = len(records); records[rec["id"]] = rec; index.add(rec["id"], rec)
top = [records[i] for i, _ in index.query("我護照咧", k=5)]
answer = await read(ask, "我護照咧", top, now_day=4)   # {"status": "answer" | "unknown", "reply", "rec_id", "quote", ...}
```

Runnable version: `python examples/with_your_llm.py` (a scripted stand-in for the LLM), or
`python examples/with_your_llm.py --openai <base_url> <model>` for an OpenAI-compatible server (that path was not
used in the research). Only the listener and reader calls need a model. The research used an LLM server running
Qwen3.8-Flash-Next; other models were not measured.

## Repository layout

| Path | What |
|---|---|
| `src/aria_memory/` | the library |
| `tests/`, `tests/fixtures/` | tests; the two 21-day scripts with their transcripts and the research's cached listening output; the 21-day always-on timeline; the reflex model's weights |
| `examples/quickstart.py` | the no-GPU tour above |
| `examples/with_your_llm.py` | listener, index and reader wired together; scripted LLM or an OpenAI-compatible server |
| `docs/research/` | paper-style write-up, every number with its source |
| `docs/REVIEW.md` | where every file came from, its licence, what was removed or renamed and why |
| `docs/VOCAB.md` | building the model vocabulary for the index |
| `docs/ENGINE.md` | the fast-weight memory inside the inference engine |
| `third_party/strata/` | the engine patch against Strata 0.1.52 (MIT, Strata's licence) |
| `docs/HUGGINGFACE.md` | proposed Hugging Face uploads (not uploaded) |
| `experiments/` | the raw research record: code, logs, result files, reports and the research log (Traditional Chinese) |

## How this differs from Letta, mem0 and Graphiti

In short: those are complete memory products for chat agents; this is research code for a voice assistant that is
always on. Here the records are never edited, retrieval is an associative memory in weights over words and sounds (no
embedding service), the reader must refuse when no record is clearly about the question, the evaluation keeps
speech-recognition errors and counts honest "I don't know" answers, and a gate bounds unprompted thinking. Table and
caveats: [docs/research/08-limitations.md](docs/research/08-limitations.md#84-comparison-with-other-memory-systems).

## Public benchmarks

LongMemEval, LoCoMo and other public long-term memory benchmarks: **not run**. How the task here differs and how one
could run them: [docs/research/08-limitations.md](docs/research/08-limitations.md#83-relation-to-public-long-term-memory-benchmarks).

## Limitations

- Synthetic data only: two 21-day Mandarin scripts, three TTS voices. Voice-ID numbers are not meaningful yet.
- Mostly single seeds; differences under about 2 questions in 51 are noise.
- One thinker model measured end to end. The always-on loop ran in a simulator, not on real devices.
- Storage tested up to 2,400 events; years of use are untested.
- The fast-weight engine experiments need the patched Strata engine (`third_party/strata/`) and a GPU.

More: [docs/research/08-limitations.md](docs/research/08-limitations.md).

## Licence

Apache-2.0 ([LICENSE](LICENSE)), except `third_party/strata/` (MIT, Strata's licence). Model weights and audio are
not included; see [NOTICE](NOTICE) and
[docs/REVIEW.md](docs/REVIEW.md).
