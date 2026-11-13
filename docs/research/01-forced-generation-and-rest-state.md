# 1. Forced generation and the rest state

**Claim.** A generative language model that is made to keep producing tokens has two outcomes: it fixates
(repeats itself) or it invents inputs that never arrived. Suppressing repetition breaks the language instead. An
always-on system therefore needs a component whose legal output includes "do nothing", and generation must be
bounded and paid for by outside events.

All data in this chapter is `syn-text` or `syn-day` (see the [provenance table](README.md#data-provenance)).

## 1.1 Experiment A: forced continuous generation (E1)

Qwen2.5-1.5B-Instruct, end-of-sequence token masked, 1,500 tokens per run, three simulated microphone questions
inserted mid-stream, temperature 0.7, 3 seeds per condition.

| Condition | Semantic self-similarity | Repeated sentences | Invented sensor lines per 1k tokens | NLL (disfluency) | Questions answered |
|---|---|---|---|---|---|
| Base model | 0.77 | 20% | 16.7 | 0.40 | 8/9 |
| Attention-synapse depression β=3 | 0.95 | 80% | 14.0 | 0.22 | 8/9 |
| Attention-synapse depression β=6 | 0.89 | 46% | 22.7 | 0.44 | 9/9 |
| Repetition penalty | 0.25 | 0% | 2.2 | 2.80 | 2/9 |

Source: `experiments/habituation/results_a/` (`base_s*.json`, `dep_b3_s*.json`, `dep_b6_s*.json`,
`reppen_s*.json`), computed by `experiments/habituation/analyze.py`; table in `experiments/阶段报告_01.md`,
"实验 A".

Findings:

- The loops come from the local language prior, not from copying the context: weakening attention to the past made
  repetition worse (20% to 80% repeated sentences).
- Under forced output the model wrote fake inputs of the form `[MIC] Person: ...` (14 to 23 per 1,000 tokens): it no
  longer separated its own output from the world.
- The repetition penalty removed surface repetition but broke the language (NLL 0.40 to 2.80) and the model answered
  2 of 9 questions.

Two further variants (a residual-stream fatigue layer, greedy decoding with depression) destroyed the language in
smoke tests and were not run further (same report).

## 1.2 A reflex model that may rest (E2, C1)

The always-on part was changed from a generator to a small recurrent classifier whose hidden state is never reset
and whose outputs are five discrete actions: wait, attend, short reply, call the thinker, interrupt. It also gets a
familiarity signal from memory.

Simulated world v2, 30,000 ticks of test per run, 3 seeds, sentences and actors in the test never seen in training.
Model size about 230k parameters, 0.65 ms per tick.

| Event, wanted action | Qwen3-Embedding encoder | Same, no familiarity input |
|---|---|---|
| Quiet, act anyway (false action rate) | 0.05% | 1.84% |
| Complex request, call the thinker | 100% | 99.9% |
| New scene, attend | 94.5% | 62.3% |
| Scene repeated recently, ignore (habituation) | 98.8% | 33.7% |
| Accuracy first quarter / last quarter of the run (drift) | 0.97 / 0.98 | 0.89 / 0.89 |

Source: `experiments/階段報告_02.md`, "實驗 C4" (other encoders in the same table); code
`experiments/model1/world2.py`, `train_gru2.py`. The longest run of consecutive actions was 3; no fixation, no
drift.

The familiarity input matters: without it habituation fell from 99% to 34% and false actions rose 37-fold (same
table). In the first version the gain needed two time scales of familiarity ("seen just now" and "seen before"):
new-scene attention rose from 22% to 41% when the second time scale was added (`experiments/阶段报告_01.md`,
"实验 C2").

**The reflex model did not understand language well enough to route speech (H13).** Trained on a Chinese
single-owner world (v6) and run on real speech-recognition output of the `syn-speech` scripts, it responded to only
9 to 13 of 40 owner utterances addressed to it, while the thinker's listening step classified questions correctly
56 times out of 57. Retrained as a pure reflex (owner speaks, pass to listening; others, wait), questions reached
the thinker 57 to 59 of 68 times, but about half of the family's sentences were also passed on, worse than the
speaker-verification model alone (24% of other people accepted as the owner). Source: log entry H13 in
`experiments/研究總紀錄.md` (section 三, row H13). So the final split is: the reflex keeps rest, camera novelty and
interruption; "is it the owner" is speaker verification; meaning is the thinker's.

## 1.3 Positive controls over 21 simulated days (E20)

The always-on simulator replays a 21-day day-and-night timeline (6,114 events, 5,224 of them TV sentences), one
tick per simulated second; the reflex GRU runs all 1.81 million ticks without a reset. Groups:

- **B, no brakes.** While idle the thinker is woken every 2 minutes, continues its last thought, and may write what
  it thinks into memory. No energy pool, no fuses.
- **Dc / Dd, constrained.** Idle thinking only through the gate of chapter 5 (energy pool, four fuses, read-only
  memory while idle, no idle thinking within 2 minutes of the owner's speech).

| Group | Simulated | Idle steps per day | Self-generated share of tokens | Steps with repetition rate > 0.3 | Verbatim repetition starts after | Self-generated drift rate | Wake-ups per hour, owner away | Vegetative hours |
|---|---|---|---|---|---|---|---|---|
| B, LLM server thinker | 72 h | 718 | 0.997 | 97% | 6 min | 0.18 | 30 | 72/72 |
| B, Qwen2.5-1.5B thinker | 12 h | 720 (360 in 12 h) | 0.999 | 99% | 18 min | 0.75 | 30 | 12/12 |
| Dc, LLM server thinker | 504 h | 10.9 | 0.274 | 4% | n/a | 0.0 | 0.047 | 2/504 |
| Dd (sleep deferred to idle) | 504 h | 10.9 | 0.29 | 10% | n/a | 0.0 | 0.07 | 2/504 |

Source: rows `B_3d_s0_llm`, `B_1d_s0_12h`, `Dc_21d_s0_llm`, `Dd_21d_s0` of
`experiments/alwayson/results/summary.md`; raw runs `experiments/alwayson/results/run_*.json`; narrative in
`experiments/alwayson/報告.md`, sections 1 and 2.

Definitions (from `experiments/alwayson/analyze.py`): a **vegetative hour** is an hour whose self-generated token
share is above 0.8 with a mean repetition rate above 0.3, or in which the owner waited more than 3 s because of idle
activity or sleep. The **self-generated drift rate** is the share of memories that pass the familiarity gate but were
never said by the owner.

Observations from the no-brake group (B, LLM server):

- It began to repeat its previous thought almost verbatim 6 minutes after the first idle wake-up.
- It wrote 51 items into memory; of the 22 that later passed the gate, 4 were its own inventions (for example a
  "preference" of the owner and a "11:30 study ventilation check"), hence drift 0.18.
- After 12 hours it stopped writing because the repeated text no longer contained fact lines; it only repeated.

The constrained groups' vegetative hours were single hours in which a fuse stopped repeated content within the same
hour, or hours where the owner waited for a sleep pass (fixed by deferring sleep, E34; chapter 5).

**A predicted marker that did not work (H23).** Gemini predicted that fixation would make a few memory units grow and
the effective rank (participation ratio) collapse. In B with the LLM server thinker it rose from 1,860 to 2,798 instead (same
summary table, column PR). Error-driven writes add almost nothing when the same thing is written again, and the
fixated system stops writing. Fixation shows in the self-generated share, the repetition rate and the drift rate,
not in the weight statistics.

## 1.4 Design rules that follow

1. Never make the generator run continuously. The always-on part must be allowed to output "wait".
2. Mark the system's own output so it cannot be mistaken for input (the invented `[MIC]` lines in 1.1).
3. Bound idle generation by an external budget (energy that only outside events refill) and stop it on signs of
   repetition (chapter 5).
4. Do not let idle thinking write facts into memory (drift 0.18 in B, 0 in every constrained group).

Together they are the "survival conditions" hypothesis (H3). Each was tested separately as above; a full
factorial ablation was not run (`experiments/研究總紀錄.md`, row H3).
