# 6. Listening under messy input

**Claim.** Once memory storage was reliable, almost all remaining losses were in hearing: speech recognition errors,
missed owner voice, and the listening step failing to cut new facts out of a stream of corrections, pronouns and
asides. Rules that do not need a model (relative dates, false-fact cleaning) and prompt changes that keep the source
sentence helped; extra cleverness on the reader side mostly did not.

Data: `syn-speech`. The scripts were written to be messy on purpose (several facts per sentence, one fact over
several sentences, corrections, pronouns across sentences, relative dates, overheard speech, TV and family
interference, never-told and similar-but-different questions).

## 6.1 Test realism

Midway through, a review found that the test inputs had been simplified to suit the model. The audit
(`experiments/研究總紀錄.md`, section 五, items T1–T13) lists every simplification and its likely effect, for example
clean template sentences with speaker labels, English only, one fact per sentence, rewritten questions, and
implausible object places ("notebook in the rice jar"). From then on the rule was: when a test fails because the
input is messy, fix the system, not the input; preprocessing counts as part of the system and is tested on raw input.
All Chinese results in this repository follow that rule.

## 6.2 The first honest measurement (E8)

21 days, 200 audio segments, 90 facts, 72 questions; Windows OneCore voices for owner, family, visitor and TV;
pink noise, clatter, TV overlapping the next sentence. Pipeline: faster-whisper, owner voice verification, the
thinker's listening step, memory, inline recall.

| Outcome of 51 told questions (best version, judged by hand) | Share |
|---|---|
| Right | 37% (19) |
| Confidently wrong | 18% (9) |
| Says it does not remember | 29% (15) |
| Owner's voice not recognised, never processed | 16% (8) |

All 21 never-told or similar questions were answered honestly. Source: log entry 2026-11-05 "第二級 v1/v2" and the
table that follows it; raw `experiments/llm_fastmem/results/zh_memory_result*.json`.

## 6.3 Hearing

| Change | Result | Source |
|---|---|---|
| faster-whisper small to large-v3-turbo with the wake word as hot word, per-word speaker split | owner character error rate 69% to 18% (median 9%) | log entry 2026-11-05 "語音辨識改用 large-v3-turbo" |
| Owner verification (resemblyzer), threshold from week 1, tested weeks 2–3 | owner recognised 83%; other voices accepted as owner 12 of 50 | same entry |
| Short-utterance voice continuation (H39) | rescued 145 owner segments, 4 wrongly accepted, but only 1 more of 102 told questions (most rescued pieces were wake-word fragments) | `experiments/p8/eval_voice_smooth.txt`; map H39 |
| ASR 3-best candidates fed to index and reader (H39) | right 67 to 68, wrong 9 to 10 (both scripts) | `experiments/p8/eval_h39b_nbest.txt` |

Synthetic voices are too few for voice-ID numbers to mean much (three voices; another synthetic male voice scored
0.91 against the owner). Real recordings are the open step (P2, chapter 8).

## 6.4 Listening

The listening step is one short thinker call per owner sentence. Prompt: `LISTEN_SYS` and `STAGE_SYS` in
`src/aria_memory/prompts.py`; code: `src/aria_memory/listen.py`.

| Change | Result | Source |
|---|---|---|
| Topic state (what is being talked about) for pronouns | part of the E24 configuration | log, Gemini round 8 |
| Staged commit: a new fact waits 2 more owner sentences, so a correction or detail replaces it (E24) | fact coverage 61% to 64.4%; with the record log 67.8%, recall 39.2% (best of that period), stale answers 0/18 | map E24 |
| Staging length 1 / 2 / 4 sentences (E32) | recall 35.3% / 39.2% / 29.4%; single runs, differences of 2–5 questions | map E32 |
| Relative dates by rule, not by the model (H29) | rule 29/29 on both scripts' gold answers; the thinker alone about 70%; end to end no change, the limit was names misheard | map H29; `experiments/p8/reldate.py` |
| Night re-listening in Chinese with the source sentence number (H30) | invalid labels 30 to 1 and 20 to 0; right label and right content 35 to 64 and 28 to 54 of 90; relative times 4 to 8 and 2 to 5 of 9 | `experiments/p8/eval_relisten.txt` |
| Send every question line to the reader (H32, part 1) | +4 and +3 told questions reach the reader | `experiments/p8/eval_listen_route.txt` |
| Drop listening output that is a question or a comment (H40) | of 3,655 writes, 17 dropped and 18 trimmed; none of them had ever been used for an answer | map H40; `experiments/p8/value_clean.py` |

E43 combined H30, H31b and H32 part 1: facts written 46.7% to 56.7% (set2) and 64.4% to 68.9% (set_old), relative
times on set2 1/9 to 5/9, invalid night labels 13 and 17 to 0 (log entry 2026-11-06 E43).

What the listening step still gets wrong (set2, every miss read by hand; log entry 2026-11-06 "聆聽寫錯的類型"):
subject and object swapped, a relative date one week off, content leaking from the previous sentence, an object name
written as a homophone. All nine "not written" misses had in fact been heard and written, wrongly.

## 6.5 Reader-side changes that did not pay

| Change | Result | Source |
|---|---|---|
| Reader told that sound-alike names count (H31) | +4 and +3 right, but never-told answered 0 to 3 on set2 | map H31 |
| Second read only when a record name has exactly the question's pinyin (H31b, adopted) | +1 and +2 right, never-told unchanged | map H31; `experiments/p8/eval_rr42_*_phon.txt` |
| A "哪/拿" (where/take) rule to catch misheard questions (H32 part 2) | chit-chat taken as questions 7 to 14 and 12 to 15; not adopted | `experiments/p8/eval_listen_route.txt` |
| Two previous sentences given to the reader (H33) | set_old 32 to 30, set2 38 to 39, never-told 2 to 0; only with pronouns (H33b): 31/6/0 vs 32/6/1; not adopted | `experiments/p8/eval_rr43_*.txt` |
| Ask back with the index's best guess when the reader says "don't know" (H37) | the top record was right in 4 of 16 such cases; asks would be wrong more often than right | map H37; `experiments/p8/eval_index_h37.txt` |

## 6.6 The wake word

A product decision limited transcription to speech after the wake word, plus an 8-second follow window after each
answer. On the scripts, with the wake word "Aria" and speech grouped by pauses of at most 1 s:

| Script | Condition | Told questions heard (of 51) | Facts heard (of 90) |
|---|---|---|---|
| set_old | always listening | 50 | 88 |
| set_old | wake word only | 24 | 24 |
| set_old | wake word + 8 s | 44 | 55 |
| set2 | always listening | 51 | 88 |
| set2 | wake word only | 31 | 45 |
| set2 | wake word + 8 s | 51 | 76 |

Source: `aria_memory.replay.coverage`; `python examples/quickstart.py` prints the set_old rows and
`tests/test_replay.py::test_coverage_under_the_wake_word` checks them. Wake-word-only listening loses about half
of the memory; the 8-second window recovers most of it; what the owner tells other people is never heard.

One replay through the full product pipeline (product side, not re-run here; public summary entry 2026-11-09 of the
log): set_old, wake word only, 7 right and 5 wrong among the told questions it heard, 33 of 72 questions heard,
20 of 21 never-told held, median turn 13.6 s with thinking on.
