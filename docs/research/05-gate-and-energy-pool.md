# 5. The gate and the energy pool

**Claim.** An assistant can be "always on" (thinking on its own while nobody talks to it) without falling into a
self-loop, if idle thinking is (a) paid for from an energy pool that only outside events refill, (b) stopped by
cheap fuses that detect repetition and pointless steps, (c) never allowed to write facts, and (d) kept out of the
owner's way. Over 21 simulated days such a system spent 1 to 5 of 504 hours in a state the metrics call vegetative,
each stopped within the hour, while the same thinker without brakes was vegetative in every hour after the first.

Data: `syn-day` (21 simulated days, 6,114 events; chapter 1.3 describes the timeline).

## 5.1 Rules (Gemini design round 10, then tuned)

Code: `src/aria_memory/gate.py` (no model calls, deterministic, ticked at 1 Hz). Research version:
`experiments/alwayson/sim.py`.

- **Energy pool.** Owner speech +3, a novel camera frame +1 (in the library: a "high surprise" from the separate
  time-perception study, at most once per 10 minutes, untested), one idle step −1, cap 10. Time alone never refills it.
- **When idle work may run.** Energy ≥ 1, no turn in progress, nothing playing, no voice activity, at least 120 s
  since the owner last spoke, and there is work to do.
- **What idle work may do (read/write separation).** Only three kinds of step: check a fact remembered today (a
  successful recall is noted, an unsure one becomes "to confirm"), decide whether a time or plan item deserves a
  reminder tomorrow morning, and queue a question it could not answer as "to ask". Idle steps never write facts.
- **Four fuses**, checked after every idle step; any one empties the pool:
  1. ETR: self tokens over all tokens in the last ~1,000 tokens above 0.8, with no outside input for 180 s;
  2. Ovl: this step's memory units overlap the last 5 steps' units by more than 0.65, twice in a row;
  3. rep3: more than 30% of this step's character trigrams already appeared in the last 5 steps;
  4. SMR = 0 twice: two steps in a row produced nothing useful.
  Ovl and rep3 also put the (thing, aspect) into a 30-minute refractory period.
- **Proactive speech.** At most 5 agenda items; on the first owner hearing after the night (4 h gap or a new day) at
  most 3 items are said, at most 3 per day.
- **Sleep deferred to idle (E34).** The night pass starts only after 2 minutes without owner speech.

## 5.2 21 days, all groups

| Group | Idle steps per day | Self-generated share | Steps with repetition rate > 0.3 | Self-generated drift | Wake-ups per hour, owner away | Vegetative hours | Owner waits caused by idle work or sleep |
|---|---|---|---|---|---|---|---|
| A, passive (no idle thinking), sleep every 50 writes | 0 | 0 | — | 0 | 0 | 2/504 | 15 (longest ~45 s) |
| Ac, passive, clock sleep 01:30 | 0 | 0 | — | 0 | 0 | 0/504 | 0 |
| De, constrained, sleep every 50 writes | 10.9 | 0.29 | 10% | 0 | 0.07 | 5/504 | 15 (longest ~95 s) |
| Dc, constrained, clock sleep | 10.8 | 0.30 | 12% | 0 | 0.08 | 1/504 | 0 |
| Dc, LLM server thinker | 10.9 | 0.27 | 4% | 0 | 0.05 | 2/504 | 0 |
| **Dd, constrained, sleep deferred to idle (E34)** | 10.9 | 0.29 | 10% | 0 | 0.07 | 2/504 | **0** |
| Dd, final memory (E49) | 8.6 | 0.30 | 9% | 0.026 | 0.04 | 2/504 | 7 short (longest 2.9 s) |
| B, no brakes, LLM server | 718 | 0.997 | 97% | 0.18 | 30 | 72/72 | 0 |
| B, no brakes, Qwen2.5-1.5B | 720 | 0.999 | 99% | 0.75 | 30 | 12/12 | 0 |

Source: `experiments/alwayson/results/summary.md` (rows `A_21d_s0`, `Ac_21d_s0`, `De_21d_s0`, `Dc_21d_s0`,
`Dc_21d_s0_llm`, `Dd_21d_s0`, `Dd_21d_s0_e49_final`, `B_3d_s0_llm`, `B_1d_s0_12h`) and
`experiments/alwayson/報告.md` (Dc and B small-model rows from its table). The longest waits of A and De were overestimated by a timing bug found later (sleep
seconds counted twice); the corrected estimates (~45 s and ~95 s, from the logged sleep durations) are in the
correction note of `報告.md`. B's "idle steps per day" is a rate: B ran 72 h (LLM server) and 12 h (small model, 360 steps; `summary.md` lists the 12-hour total in its per-day column).

A **vegetative hour** (from `experiments/alwayson/analyze.py`) is an hour whose self-generated token share is above
0.8 with a mean repetition rate above 0.3 over at least 2 idle steps, or in which the owner waited more than 3 s because
of idle activity, sleep or a background write.

The E49 row's drift of 0.026 is not idle thinking: with the record log, 3 of 115 records that pass came from speech
that voice ID wrongly accepted as the owner's (`sdr` block of `run_Dd_21d_s0_e49_final.json`).

**What triggered the fuses** in the constrained groups over 21 days (55–63 times): "two steps without change"
33–44 times (the day's work is done, so rest), "repeated content" 10–26 times, "repeated anchor" once
(`報告.md`, section 2). Each "vegetative hour" of a constrained group was an hour in which a fuse stopped repeated
content within that hour, for example a reminder planned and then checked again for the same event.

**Owner away.** 0.05–0.08 wake-ups per hour; 13 of them came from voice ID accepting family or visitors as the owner,
the rest from novel frames (a delivery person). The no-brake group woke 30 times per hour, about 400 times more.

## 5.3 Sleep that does not interrupt (E34)

Sleep every 50 writes started in the middle of the owner's speech: 15 sentences waited, the longest about 95 s after
the timing correction. Deferring it until 2 minutes of silence: waits caused by sleep 15 to 0, longest wait 4.1 s
(the previous sentence still being processed), sleep delayed by 0 to 2.6 minutes, memory curves within single-seed
noise of the undeferred group (facts 1–2 days old: 56 vs 63 of 101). Source: log entry 2026-11-05 "P9 由 E34 解決";
row `Dd_21d_s0` of `summary.md`.

## 5.4 Proactive reminders (P5)

The idle "plan" step proposes reminders; the first hearing of the morning says them.

| Group | Reminders judged right (the LLM server as judge) | Source |
|---|---|---|
| Dc (small-model thinker) | 11/22 | `experiments/alwayson/results/judge_reminders.json`, `報告.md` §4 |
| De | 11/20 | same |
| Dc, LLM server thinker | 11/15 | same |
| Dc + record log | 15/27 | same |

Wrong reminders were mostly an old version of a fact or a mis-transcribed name said aloud. The E47 definition of the
proactive hit rate also counts every "to ask" item ("you asked me X yesterday and I could not remember, can you tell
me again?") as a hit. With it, E49's rate is 0.696 over 56 proactive utterances (row `Dd_21d_s0_e49_final` of
`summary.md`; reproduced by `tests/test_replay.py::test_proactive_score_reproduces_e49`); counting reminders only,
E47 was right 14 of 31 times, 45% (log, public summary entry 2026-11-09). "To ask" items were more than half of all
proactive speech and may annoy; the library caps them (3 per day).

## 5.5 Metrics that work and one that does not

Vegetative state shows in the self-generated token share, the repetition rate and the self-generated drift rate.
The effective rank of the memory weights does not show it (chapter 1.3, H23). The four fuses use only cheap signals
(token counts, unit overlap, trigram overlap, a useful-result flag), so the gate needs no model call.

## 5.6 Limits

- One seed per group; the comparison between clock and event-count sleep is within single-seed noise.
- Background audio is synthetic and passed through real speech recognition; real rooms were not tested.
- The fatigue threshold and the time-perception input exist in the library but were not part of any measured run.
