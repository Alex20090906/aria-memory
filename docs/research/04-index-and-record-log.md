# 4. The index and the record log (the final design, E49)

**Claim.** For long-term memory of spoken life, split the job three ways. An **append-only record log** keeps every
fact with the sentence it was heard in, never edited. A **sparse associative memory in network weights** (the
"hippocampal index") maps the words and sounds of each record to the record's position and, given a question,
proposes the five most likely records. The **thinker reads** those records and answers only when one is clearly about
the same thing and aspect. Retrieval by the network was as accurate as an external text-embedding model at small
scale and more robust with 10,000 hard distractors; the end-to-end system answered 30 of 51 told questions right,
6 wrong and held 20 of 21 never-told questions on synthetic speech, with a median reading time of 1.54 s.

Data: `syn-speech` (two 21-day scripts: set_old, written first, and set2, a candle-studio owner's life, both with
51 told questions and 21 never-told or similar questions) and `syn-day` for the 21-day always-on runs.

## 4.1 From labels to a record log

The hypothesis tested here: the network stores only a label that opens an exact record in an append-only log, like a
diary (compare the hippocampal indexing theory, Teyler and DiScenna 1986). Chapter 3 shows why: content in the log cannot be worn
down (E28, E39: 56/60, 0 confidently wrong after 2,400 events). In the 21-day always-on simulator the same change
took facts told 14–20 days earlier from 0/11 to 11/11 and 7–10 days earlier from 3 to 39 of 41 (E33; log entry
2026-11-05, E33 row of the always-on table and its item 5, raw `experiments/alwayson/results/run_Dc_21d_s0_log.json`).

The cost: the problem moves to **finding the right label** from messy speech (P8). With exact labels 20/51 told
questions were answered (E36). Every label-matching method hit a ceiling:

| Method | set_old | set2 | Source |
|---|---|---|---|
| Exact label | 20/51 | 12/51 | map E36, E37 |
| Semantic label search (Qwen3-Embedding cosine ≥ 0.6) + ask back when unsure | 26/51 | 17/51 | map E37; `experiments/p8/label_finder_eval.py` |
| + pinyin sound score (H25) | 28 to 28 | 17 to 18 | map H25; `experiments/p8/eval_phon_*.txt` |
| Merge similar labels at write time (H27) | 28 to 26 | 17 to 16 | map H27; `experiments/p8/write_merge_eval.py` |

Typical misses on set2: speech recognition wrote the object name as a homophone, the right object was stored under
the wrong aspect, or a relative date was not converted (log entry 2026-11-06 E37 新腳本).

## 4.2 Retrieve the sentence, then read (H28)

Instead of matching labels, keep the original sentence with every record, find the 5 records closest to the
question, and let the thinker read them with strict rules (use a record only if it is about the same thing and aspect;
a similar thing is not enough; newest day wins; otherwise say you don't remember). Prompt: `READ_SYS` in
`src/aria_memory/prompts.py`.

- Offline on the same listening output: set2 27/51, set_old 34/51 (map H28).
- End to end on the research machine (E42): set_old 30, set2 25 (previous best 26 and 19); never-told answered
  1/21 and 0/21 (log entry 2026-11-06 E42).

Three fixes on the listening side then followed (chapter 6 has the details): a Chinese night re-listening format
with the source sentence (H30), a second read when a record's name sounds exactly like a word in the question
(H31b), and sending every question line to the reader (H32, first half). End to end (E43): **set2 25 to 38,
set_old 30 to 31**, never-told answered 0/21 and 1/21 (log entry 2026-11-06 E43; raw
`experiments/llm_fastmem/results/zh_result_I_all_old.json`, `zh_result_I_all_set2.json`).

## 4.3 The network finds (H35, H41)

Gemini's round-11 suggestion: let the network do the finding. Each record gets a fixed code (one row of a random
orthogonal matrix, 512 dimensions); the key of a text is a sparse activation over 65,536 units:

- **meaning units** (0–32,767): each model token's layer-0 word vector, normalised and projected with a fixed random
  matrix; its 4 strongest units;
- **sound units** (32,768–65,535): every toneless pinyin syllable and syllable pair (zh/z, ch/c, sh/s, ang/an,
  eng/en, ing/in merged), 2 hashed units each.

Units are weighted by 1/√(n_i + 10) where n_i counts the records that used unit i; the 128 strongest are kept. Write
by the delta rule `W += z (c − 0.2 Wᵀz)ᵀ`; read `y = Wᵀ z(question)` and rank records by `c · y`. Code:
`src/aria_memory/index.py` (pure numpy; W is 65,536 × 512 float32, 128 MB).

| Check | set_old | set2 | Source |
|---|---|---|---|
| Right record in top 5, network index (random codes, seed 7, dim 256) | 33/38 | 35/38 | `experiments/p8/eval_index.txt` |
| Same, external dense text embedding | 35/38 | 34/38 | same file |
| Right record in top 5, orthogonal codes, seeds 7, 9, 10 (H41) | 35/38 each | 34/38 each | `experiments/p8/eval_index_read512*.txt`, `eval_h41_ortho.txt` |
| Random codes over 6 seeds (mean) | 33–35 (34.0) | 33–35 (34.2) | map H35 (correction note) |
| With 10,000 same-name hard distractors: network / text embedding | 18 / 15 | 29 / 14 | `experiments/p8/eval_index_scale_dense.txt` |
| With 50,000 distractors (network only) | 14 | 24 | `experiments/p8/eval_index_scale.txt` |
| Save at half, reload, write the rest: same top 5 for 90/90 questions, W bit-identical | | | map H35; `experiments/p8/index_persist_test.py` |
| **CharVocab (no model: characters hashed, plus the same sound units)** | **34/38** | **34/38** | `python examples/quickstart.py` (added 2026-11-13) |

The last row is new in this repository: without any model vocabulary the index reached nearly the same recall on
these scripts, so most of the retrieval comes from exact characters and their sounds; the layer-0 meaning units added
one question on set_old.

The readout strength is **not** a reliable "have I heard this" gate (AUC 0.72–0.83, same files): deciding whether
the memory knows the answer stays with the reader.

## 4.4 Always-on, 21 days, strict scoring (E44–E49)

All runs: the 21-day always-on simulator with sleep deferred to idle (Dd), the LLM server as thinker, idle-time writes,
listening from E43's cached output of set_old. Strict scoring: numbers and weekdays normalised, at least 60%
character overlap with the expected answer, and a different 第N天 (day number) is wrong.

| Run | Change | Right / 51 | Wrong | Never-told held / 21 | Read median | p90 | Max | Source |
|---|---|---|---|---|---|---|---|---|
| E41 | label log + exact labels | 18 | 9 | 20 | | | | `run_Dd_21d_s0_e41.json` |
| E41 find | + semantic label search | 23 | 13 | 18 | | | | `run_Dd_21d_s0_e41_find.json` |
| E44 find | E43 listening + label search | 29 | 10 | 18 | | | | `run_Dd_21d_s0_e44_find.json` |
| E44 read | E43 listening + retrieve-then-read (external text embedding) | 30 | 5 | 20 | 2.21 s | 3.09 | 4.68 | `run_Dd_21d_s0_e44_read.json` |
| E45 | + text matching moved to idle time | 30 | 5 | 20 | 2.19 s | 2.85 | 3.81 | `..._e45_read_idle.json` |
| E46 | network index instead of text embedding | 31 | 5 | 20 | 1.66 s | 2.41 | 3.78 | `..._e46_index.json` |
| E47 | no familiarity gate on the answer path | 31 | 5 | 20 | 1.54 s | 1.98 | 3.07 | `..._e47_nogate.json` |
| E48 | + false-fact cleaning (H40) + split meaning/sound quotas (H38a) | 29 | 5 | 20 | 1.52 s | 1.87 | 3.12 | `..._e48_full.json` |
| **E49** | **E47 + orthogonal codes (final)** | **30** | **6** | **20** | **1.54 s** | **1.90** | **3.11** | `..._e49_final.json` |

Scores: `experiments/alwayson/rescore_strict_output.txt` (from `rescore_strict.py`); times:
`experiments/alwayson/latency_table.txt` (from `latency_table.py`, reading `mem_calls.read_s`). Files are in
`experiments/alwayson/results/`. Index query 13 ms, index write 36 ms (medians, same table).

Notes:

- The familiarity gate never vetoed a read in 508 reads (E42–E46): records and network were written together and the
  network did not decay, so it always passed. Removing it changed no answer and saved 0.12 s (E47; log entry
  2026-11-06). If deliberate forgetting is added back, the gate must move to write time.
- E48's two Gemini suggestions had no measurable effect; the 2-question drop is within the ranking noise of code
  reassignment (seed range 33–35 in 4.3).
- E49 differs from E47 by one question, within the reading noise; it was kept because orthogonal codes make retrieval
  independent of the seed for the first 512 records.
- The same answers re-scored by this repository's library: `tests/test_replay.py::test_strict_rescoring_reproduces_e49`
  (30, 6, 20).

## 4.5 Marking replaced records at night (H42)

Records are never edited, so "the passport moved" leaves two records. Each night the thinker could mark older records
that a new record replaces; marked records are left out of later reading.

- Loose version: 26 and 22 marks (set_old, set2); end to end set_old 31 unchanged, set2 36 to 37. Most marks were
  right; some crossed different things, different aspects, or let a false fact replace a true one
  (`experiments/p8/eval_h42_supersede.txt`).
- Strict version (only facts, same aspect, all three conditions: same entity, same aspect, new content): 19 and 6
  marks; crossings disappeared; end to end equal to no marking (`experiments/p8/eval_h42b_strict.txt`).
- Re-judged on 2026-11-09 with a written definition of a wrong mark (crosses things; hides an answer still needed and
  not re-told; replaces a current fact with something the owner never said): set_old 11 right, 2 wrong, 6 unclear;
  set2 3 right, 0 wrong, 3 unclear. The two wrong marks: a spare key "taken" (拿去, not a place) and a dentist
  appointment replaced by a teeth-cleaning one (different things). Code: `judge_mark` and `supersede_eval` in
  `src/aria_memory/replay.py`; the library adds two guards that reject both cases (`same_thing`, `value_of_kind` in
  `src/aria_memory/sleep.py`, tested in `tests/test_read_sleep.py`).

Over 21 days marking gave no measurable benefit; it is kept as a long-term step for when one object has many versions.

## 4.6 What stays where

| Part | Lives in | Changes after writing |
|---|---|---|
| Record: thing, aspect, value, day, heard sentence, source | append-only log (caller's storage) | never; a "forget" removes a record and rebuilds the index without it (`NetIndex.forget`) |
| Index W (65,536 × 512) and unit counts | network weights | only by new writes |
| Night marks (record replaced by record) | caller's storage | added at night, never removed |
| Reading rules | thinker prompt | fixed |

The original goal was "memory in the network, no external database". The final system keeps **finding** in the
network and **wording** in a log that the network alone can open; it is a diary, not a queryable database. Whether
that satisfies the goal is a judgement left to the reader; the data says the split is what made long-term memory
reliable.
