# Review: where every file came from

This file records what the public repository contains, where each part came from, under which licence, and what was
removed or changed on the way from the private research folder. It was prepared on 2026-11-13.

## 1. No real people

No file contains a real person's voice, voiceprint, life events, home or account details.

- **Speech.** All test speech was synthesised with the Windows OneCore text-to-speech voices for Taiwanese Mandarin
  (owner: Zhiwei; family: Hanhan; visitor and TV: Yating), mixed with generated noise, then transcribed by
  faster-whisper. Only the transcripts are included. The audio files (809 WAV files, about 250 MB) are not.
- **Scripts.** The 21-day household scripts were written by Gemini; the English fact lists, one-year script and
  simulated world by Gemini, Claude or templates. Names in them are invented.
- **The researcher's own recordings** were never made during the research (it paused waiting for them). The reading
  script prepared for them is `experiments/recording/讀稿_真人聲音.md`; it is text only.

### Regenerating the audio with any TTS

`experiments/realistic/render_audio.py` turns `zh_scripts.json` into one noisy recording per day. It calls
`tts(text, voice, rate, pitch)`, which uses `tts_onecore.ps1` (Windows only). Replace that one function with any TTS
that returns 16 kHz mono float32 audio, keep one voice per role (owner, family, visitor, TV), and the rest of the
pipeline (noise, gaps, TV overlap, timeline) is unchanged. Then run speech recognition and voice verification with
`experiments/realistic/hear_zh.py`. Expect different transcripts and therefore somewhat different numbers.

## 2. What was changed in copied research files

The research record in `experiments/` is copied from the private research folder with only these changes:

| Change | Where | Why |
|---|---|---|
| The LLM server's name (Strata, an MIT open-source engine; named in NOTICE and docs/ENGINE.md) replaced: next to Chinese text by 語言模型伺服器, elsewhere by "LLM server"; in identifiers, paths and file names by `llm` (for example `llm_fastmem/`, `memory_llm.py`, `run_B_3d_s0_llm.json`), and the upper-case form by `LLM` | all text files, 11 file or folder names | project naming rule for the public release |
| The owner role word replaced by 使用者 ("user") | all text files, prompts and fixtures included (6,453 occurrences) | project naming rule |
| The assistant's product name replaced by 助理 ("the assistant"); the product app's internal name removed | research log and a few scripts | project naming rule |
| Invented English test names: one that matched the product's internal name became `Yumi`; `Alex` became `Axel` | English test data and code (565 and 763 occurrences) | avoid names that match the product or the computer's user name |
| Addressing the researcher as "you" (你) changed to 研究者 outside quotes | research log, research map, reports | the log was written to the researcher |
| Lines about cloud account balances, quotas and machine operations removed (9 lines); two quoted sentences of the researcher replaced by a summary | `experiments/研究總紀錄.md` | private account details and private conversation |
| Log entries from 2026-11-09 on (integration into the researcher's own product) replaced by a public summary with the same numbers | end of `experiments/研究總紀錄.md` | product names and private decisions |
| Home paths, the research machine's network address, user account, SSH key name and private page link replaced by placeholders (`~`, `<workspace>`, `<thinker-host>`, `<user>`, `<ssh-key>`, `<thinker-workdir>`, `<engine-src>`, `<data>`, `<recordings>`, `(private page)`) | code and logs | private details |
| A name artefact fixed (`Yumi 變 umi` in report 06, the example of a dropped first letter) | `experiments/階段報告_06.md` | keep the example consistent after the rename |
| Section 二 (the researcher's conjectures U1–U18) replaced by a one-paragraph note; three changelog lines that only announced a conjecture or the design note for Gemini removed; two items of the problem list that described the researcher's theory removed. U numbers that label tested ideas elsewhere stay | `experiments/研究總紀錄.md` | the researcher's decision: conjectures not published |
| Sentences and sections that mapped results onto the researcher's theory removed or neutralised | `阶段报告_01.md`, `階段報告_02.md`–`_05.md`, `alwayson/任務說明.md` | same |
| Gemini rounds 4, 5, 8, 10, 11 and 12 condensed to one line of attribution plus what was implemented or verified here | research log, `階段報告_04.md`, `階段報告_05.md` | the researcher's decision: discussion records not published |
| The 18 conjecture nodes (U1–U18) and their links removed from the research map, which was regenerated with `map_tool.py`; round 10's node keeps one line of attribution | `experiments/map/` | same |

Because of the word replacements, the stored prompts and outputs no longer contain the exact words the research
used. The numbers are unaffected: the test suite still reproduces E49's strict score (30, 6, 20) and the index's
top-5 recall (35/38, 34/38) on the renamed fixtures. One check needed a change: the token-set comparison in
`tests/test_index.py::TestRealVocab::test_tokenizer_matches_the_research` leaves out the tokens of 使用者.

Files added for this release: `experiments/cloud/rescore_output.txt` (output of the existing
`rescore_gputw.py`), `experiments/alwayson/rescore_strict_output.txt` (output of the existing `rescore_strict.py`),
`experiments/alwayson/latency_table.py` and `latency_table.txt`, `experiments/README.md`. They read stored outputs
only; no model was run.

## 3. What was left out

| Left out | Size | Why |
|---|---|---|
| All audio (`realistic/audio/`, `realistic/set2/audio/`, `alwayson/audio/`, `sensors/testdata/`) | ~250 MB | synthetic, but made with Windows voices whose output licence is unclear; regenerate with any TTS (section 1) |
| Images (`sensors/testdata/frames/`, one timeline plot) | small | synthetic frames; not needed |
| Trained PyTorch checkpoints (`*.pt`, 32 files) of the reflex model and small-model memories | ~25 MB | proposed for Hugging Face instead (docs/HUGGINGFACE.md); the final reflex weights are in `tests/fixtures/` as numpy |
| `p8/llm_tok/`: the thinker model's tokenizer files and rows of its embedding table | ~60 MB | derived from the model's weights; its licence applies |
| `tests/fixtures/vocab_small/`: a 1,595-token sample of the model vocabulary | 57 KB | derived from the model's layer-0 table; build your own (docs/VOCAB.md) |
| Upstream source of the LLM server (`ext/`, a 30 MB tarball and its tree) | 123 MB | third-party code, available upstream |
| Cloud billing ledger, instance ids, build logs, deployment bundle, helper scripts | — | account details |
| The live research star-map page and its database version | — | private page |
| Machine operation scripts (start and stop the research server) | — | private setup |
| A draft report superseded by the final one, the research-notes folder of the literature review, two zip archives | — | duplicates |
| The reading script's later version that used the product's wake word | — | the earlier version with "Aria" is identical otherwise and is included |
| The researcher's conjectures U1–U18 (including the theory behind the research), the theory-integration note (`理論整合_01.md`) and the overall-design note prepared for a design round (`數位生命_整體設計.md`) | — | the researcher's decision (2026-11-13): not published |
| `experiments/discussions/`: prompts and replies of the design rounds with Gemini (rounds 1–12) and the small Gemini client | 656 KB | the researcher's decision: not published. Scripts that imported the client need a replacement to run |
| The Traditional Chinese literature review written at the start (it was organised around the researcher's theory) | 49 KB | same |
| `__pycache__`, the research virtual environment | — | build artefacts |

## 4. Files a person should look at

The privacy scanner lists data files it cannot read as text. Each was checked by hand:

| File | Content | Source | Licence |
|---|---|---|---|
| `tests/fixtures/reflex_gru_w6r.npz` | weights of the reflex GRU (231,686 float32 parameters, 10 arrays) | trained in this research on the synthetic Chinese world v6 (`experiments/model1/world6.py`, `train_gru3.py --world w6`, results in `model1/results_w6r/`); inputs are Qwen3-Embedding-0.6B sentence vectors | Apache-2.0 (this repository) |
| `tests/fixtures/reflex_gru_golden.npz` | 12 input vectors, outputs and final hidden state of the original PyTorch model, for the parity test | computed from the above | Apache-2.0 |
| `tests/fixtures/e49_token_ids.npy` | 1,806 integer token ids used by E49's records | tokenizer of the thinker model | integers only; no text, no weights |
| `tests/fixtures/alwayson/timeline_21d.json.gz` | the 21-day always-on timeline: speech-recognition text of the synthetic scripts and synthetic background sound, simulated camera captions | built by `experiments/alwayson/timeline_build.py` and `bg_bank.py` | Apache-2.0; text written by Gemini and Claude |
| `tests/fixtures/alwayson/reflex_21d.json.gz` | the reflex model's decisions over the timeline | computed | Apache-2.0 |
| `experiments/sensors/replay_log.jsonl` | 29 lines of a smoke test of the sensor loop on synthetic events | computed | Apache-2.0 |
| `third_party/strata/fastmem.patch`, `fmem.cu`, `fmem.hpp` (text, scanned) | the fast-weight memory patch against Strata 0.1.52 (`a42cc3d`) | the research engine repository on the research main computer (12 commits, 2026-11-04 to 11-05), squashed against the upstream files on 2026-11-13; read-only, nothing changed there; checked to apply cleanly and to reproduce the last research state | MIT (Strata's licence, `LICENSE.Strata`) |

Both `.json.gz` files were decompressed and scanned with the same rules (0 hits).

Other data in the repository is JSON, text or Markdown and is covered by the scanner: the scripts and transcripts in
`tests/fixtures/set_old`, `tests/fixtures/set2` and `experiments/`, the LLM outputs cached in `listen_cache.json` and
the `results/` folders (generated by Qwen3.8-Flash-Next under the Qwen Community License 1.0; Qwen2.5-1.5B-Instruct
for small-model runs).

## 5. Scan

`python -I tools/scan_private.py aria-memory` (the project's private checker: secrets by value and by shape,
private terms, addresses, Windows user paths): **0 hits, 6 files to review** (section 4).
