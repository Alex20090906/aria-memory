# 8. Limitations and open problems

## 8.1 Limitations of the evidence

- **Synthetic data only.** Every voice is a Windows text-to-speech voice; every script was written by an LLM;
  every room sound is generated. No real person's speech, home or life was used. Voice-ID numbers in particular are
  unreliable with three synthetic voices.
- **Two scripts, one language.** Both 21-day scripts are Taiwanese Mandarin with 51 told and 21 never-told questions
  each. Scripts in other languages, other households and longer periods were not tested.
- **Single seeds.** Most end-to-end runs are one seed. Where seeds were varied, results moved by up to 10 of 60
  (chapter 3) or 2 of 51 (chapter 4); differences smaller than that are not evidence.
- **One thinker.** All end-to-end runs used an LLM server with Qwen3.8-Flash-Next IQ3_S. Other LLMs as listener and reader
  were not measured. The library takes any `ask(messages, max_tokens)` function, so this is easy to test but untested.
- **The always-on loop is simulated.** 21 days ran in a simulator with recorded transcripts; the gate has not run on
  real microphones and cameras for days.
- **Months and years are untested.** The longest storage test is 2,400 events. The index's orthogonal codes cover the
  first 512 records exactly; after that it falls back to random codes (H41 limit: about day 78 at 135 records per
  21 days), and hard-distractor tests show recall falling with scale (chapter 4.3).
- **The engine experiments need a GPU.** The fast-weight memory inside the LLM server's engine (chapters 2 and 3)
  needs the patched engine (`third_party/strata/`) and the 78 GB thinker model; they were not re-run for this release.
  The final system (chapter 4) does not need the modified engine.

## 8.2 Open problems

| Id | Problem | State |
|---|---|---|
| P1 | Listening coverage around 60–70% of facts (implied facts, corrections and overheard speech are worst) | partly addressed (chapter 6.4) |
| P2 | Owner voice ID and ASR on real recordings; wake-word misses and false wakes | needs real recordings (a reading script is in `experiments/recording/`) |
| H43 | The assistant hears its own voice through the speaker (echo) and may write it back as the owner's words; cost of a half-duplex gate | simulation written (`experiments/p8/echo_sim.py`), no result |
| P6 | Continuous-time neuron dynamics (membrane potential, eligibility traces, mixed clocks) | not implemented |
| — | Moving the index into the thinker's forward pass (a read head after layer 0) | engineering only; same math |
| — | Deliberate forgetting with a write-time familiarity gate | designed, not tested |

## 8.3 Relation to public long-term memory benchmarks

**Not run.** No number in this repository comes from LongMemEval, LoCoMo or any other public benchmark, and none
should be inferred from these results.

How they differ from what was measured here, to help anyone who runs them:

- LongMemEval and LoCoMo are text conversations between two parties, mostly in English, with questions over long
  chat histories. This work's input is noisy transcribed speech in Mandarin from one owner amid TV and family, and
  most of its losses are in hearing.
- The questions here are mostly "where is / when is / what do I prefer", with updates over time, and the scoring
  separates right, confidently wrong and honest "I don't know", with 21 never-told questions per script.
- To run a text benchmark, feed each user turn to the listener (`aria_memory.Listener.hear`) as owner speech, store
  the records, index them (`NetIndex.add`) and answer each question with `aria_memory.read`. The listening prompt is
  Chinese and household-specific; for English chat it would need translation and probably new few-shot examples.

## 8.4 Comparison with other memory systems

Descriptions of the other systems summarise their public documentation as known up to mid-2026 (not re-checked for
this release) and are deliberately brief; no head-to-head experiment was run.

| | This work | Letta (MemGPT) | mem0 | Graphiti (Zep) |
|---|---|---|---|---|
| What is stored | append-only records (thing, aspect, value, day, heard sentence) | memory blocks the agent edits, plus archival search | extracted facts that are added, updated or deleted | a temporal knowledge graph of entities and facts with validity times |
| Who decides what to remember | a short listening call per owner sentence, with a 2-sentence correction window and a night re-listen | the agent itself, through tools | an extraction step over the conversation | an ingestion pipeline over episodes |
| Retrieval | a sparse associative memory in weights over words and sounds (no embedding service), then a strict reader | agent tool calls (search over archival memory) | vector and optional graph search | hybrid semantic, keyword and graph search |
| Updates | never edit; newer records rank with older ones; optional night marks | the agent rewrites blocks | update or delete operations | invalidates old edges with timestamps |
| Always-on behaviour | an energy-pool gate decides when to think unprompted; measured over 21 simulated days | not the focus | not the focus | not the focus |
| Focus of evaluation | messy speech, honesty on never-told questions | long agent sessions | chat memory benchmarks | temporal reasoning over conversations |

The distinctive pieces here are the speech-first evaluation (transcription errors kept), the honesty measure on
never-told questions, the network-weight index that needs no embedding model, and the bounded always-on loop. The
other systems are far more complete as products; this repository is research code.
