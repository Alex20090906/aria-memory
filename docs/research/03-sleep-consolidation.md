# 3. Sleep consolidation

**Claim.** When facts live inside a fixed-size fast-weight memory, sleep consolidation (replay what is familiar and
was asked about, promote what the model completes correctly to slow decay, shrink the rest) beats "never forget"
under moderate load. Under heavy load both fail, and sleep fails in the worst way: familiarity survives while the
content is overwritten, so the oldest facts are answered **confidently and wrongly**. Moving the content out of the
weights into an append-only log, with the network only recognising what it has heard, removes the problem
(56/60 right, 0 confidently wrong, after 2,400 events).

Data: `syn-text`. Each run interleaves "core" facts (one person, one object, one place) with one-off noise events,
asks every core fact at the end, and splits the answers into **right**, **honest forget** (says nothing) and
**confidently wrong**. From E18 on, places are 5 plausible places per object (48 objects), and a no-memory guess
baseline is reported (0/60 in every run: the model never guesses a place right without memory).

## 3.1 The rule

Every unit of the fast-weight memory has a familiarity θ, a decay time constant τ and a recall count c. Every 50
writes the system sleeps: it picks events that are familiar and were asked about, lets the thinker complete their
anchor (self-generated replay), and a plausibility check rejects completions that do not look like a place. Units of
correct completions are promoted from τ = 100 to τ = 500; all other units are multiplied by 0.1. Source:
`experiments/llm_fastmem/sleep.py` (copied into `experiments/cloud/` bundles for the cloud runs), described in
`experiments/階段報告_06.md`, section 二.3.

## 3.2 All runs

"Ages" are recall by age quartile, youngest first. Runs from E18 on were re-scored with case-insensitive matching by
`experiments/cloud/rescore_gputw.py`; its output for every logged file is `experiments/cloud/rescore_output.txt`
(produced 2026-11-13 from the stored answers, no model calls).

| Id | Load | Memory mode | Right | Honest forget | Confidently wrong | Ages (young to old) | Source |
|---|---|---|---|---|---|---|---|
| E6 | 20 core + 180 noise | fast decay / never forget / **sleep** | 2 / 20 / **19** of 20 | | | | `llm_fastmem/results/sleep_sleep_n180.json`; report 06 |
| E6 | 20 core + 480 noise | fast decay / never forget / **sleep** | 2 / 16 / **17** of 20 | | | | `llm_fastmem/results/sleep_sleep_n480.json`, `sleep_never_n480.json`; report 06 |
| — | 40 core + 560 noise | sleep | 34/40 | 0 | 6 | 9, 9, 8, 8 | log 2026-11-05 "長期遺忘曲線" (raw file not kept) |
| — | 40 core + 560 noise | never decay | 26/40 | 6 | 8 | 7, 8, 6, 5 | same |
| E10 | 60 + 1,140, D = 16,384 | sleep | 22/60 | 5 | **33** | 12, 8, 2, 0 | same entry |
| E10 | same | never decay | 18/60 | 38 | 4 | 13, 3, 1, 1 | same |
| E11 | same, D = 65,536 | sleep | 26/60 | 19 | 15 | 13, 8, 5, 0 | log 2026-11-05 "放大到 D=65536" |
| E11 | same | never decay | 11/60 | 45 | 4 | 10, 1, 0, 0 | same |
| E12 | E11 + first-token confidence ≥ 0.5 | sleep | 19/60 | 34 | 7 | | map row E12 |
| E13 | E11 + familiarity decays with content | sleep | 13/60 | 45 | 2 | | map row E13 |
| E18 | 60 + 1,140, plausible places | sleep | 34/60 | 12 | 14 | 14, 11, 8, 1 | `cloud/results_gputw/real_sleep__*.json`; rescore |
| E18 | same | sleep + echo familiarity | 35/60 | 10 | 15 | 14, 11, 9, 1 | `real_sleep_echo__*.json` |
| E18 | same | never decay | 14/60 | 42 | 4 | 11, 3, 0, 0 | `real_never__*.json` |
| E18 | same | never decay + echo | 50/60 | 4 | 6 | 11, 12, 14, 13 | `real_never_echo__*.json`; **7/60 never-told questions answered** |
| E19 | same | sleep + echo, consolidated rows frozen | 29/60 | 11 | 20 | 13, 10, 6, 0 | `cloud/results_modal/sleep_sleep_n1140_echo_protect.json` |
| E19 | same | sleep + echo, weakest token ≥ 0.9 applied while running | 7/60 | 52 | 1 | 4, 3, 0, 0 | `..._echo_pmin09.json` (the threshold also blocked sleep replay) |
| E19 | same, seed 1 | sleep + echo | 36/60 | 3 | 21 | 14, 12, 6, 4 | `..._echo_seed1.json` |
| E27 | same | sleep + echo, soft protection α = 0.25 | 30/60 | 11 | 19 | 13, 10, 7, 0 | `..._soft_protect25.json` |
| E29 | same | sleep every 25 events | 37/60 | 12 | 11 | 14, 12, 10, 1 | `cloud/results_gputw/sleep_every25__*.json` |
| E30 | same, seed 2 | sleep | 26/60 | 10 | 24 | 13, 11, 1, 1 | `seed2_sleep__*.json` |
| E35 | same | sleep, with / without echo, never-told probe | 35/60 both | 11 | 14 | 14, 11, 9, 1 | `probe_sleep__*.json`, `probe_sleep_echo__*.json`; never-told answered 1/60 and 2/60 |
| E31 | 60 + 2,340 | sleep | 15/60 | 18 | 27 | 11, 2, 1, 1 | `load2400__*.json` |
| **E28** | 60 + 1,140 | **content from the record log**, network only as familiarity gate (sleep mode) | **56/60** | 4 | **0** | 14, 13, 14, **15** | `cloud/results_modal/sleep_sleep_n1140_u15_log.json` |
| **E39** | 60 + 2,340 | record log + never decay + echo gate | **56/60** | 4 | **0** | 14, 13, 14, **15** | `cloud/results_modal/sleep_never_n2340_u15_never_echo_2400.json`, `e39_stdout.log`; never-told answered 0/60 |

Paths are relative to `experiments/`. "Report 06" is `experiments/階段報告_06.md`.

## 3.3 Reading the table

1. **Moderate load: sleep wins.** At 480 noise events sleep kept 17/20 against 16/20 for never-forget; at 600
   events 34/40 against 26/40, and old facts barely faded (9, 9, 8, 8).
2. **Heavy load: sleep fails dangerously.** At 1,200 events in 16,384 units, sleep left 33 of 60 answers confidently
   wrong and 0/15 of the oldest right. Promoted slow units only accumulate; the familiarity of an old fact survives
   while its content is overwritten. Never-decay at least forgets honestly (38 honest, 4 wrong). This is the
   engineering version of a confident false memory.
3. **Size, confidence and protection do not rescue it.** Four times more units halved the confident errors (33 to
   15) but left the oldest quarter at 0. Freezing consolidated rows (29 right, 20 wrong) or protecting them softly
   (30, 19) was worse than no protection (35, 14). Sleeping twice as often gave 37 against 35 and the oldest quarter
   stayed at 1/15. Seeds alone moved the result by about 10 (26, 35, 36).
4. **A weakest-token confidence check turns most confident errors into honest ones** (E18 sleep: 34/14 becomes 31/0
   at threshold 0.5; same rescore file), but it must only be used at answer time: applied while running it also
   blocks sleep replay (E19, 7/60).
5. **Echo familiarity makes never-decay strong but invents.** 50/60 right with the oldest quarter 13/15, but 7 of 60
   never-told questions got an answer, mostly with high confidence: the answer was the true content of a similar
   name (Tara's earrings answered with Sara's place, E35). A confidence check cannot see this error (H22).
6. **Separating "have I heard this" from "what exactly was said" solves both.** When the network only gates and the
   words come from an append-only record log opened by an exact label, the content cannot be overwritten: 56/60 right,
   0 confidently wrong, oldest quarter 15/15, both at 1,200 and at 2,400 events, and 0/60 never-told questions
   answered at 2,400. The 4 honest forgets are all one write bug (H24: the object name appeared again inside the
   answer, so the write key landed on the wrong token; fixed and verified on the research engine, E40). The
   remaining problem moves to finding the right label from messy speech, chapter 4.

## 3.4 What carried over

The final system (chapter 4) keeps familiarity and content apart, but puts content in a log that never decays;
the network's job becomes indexing. A dense slow matrix inside the model failed (chapter 2.6), so any slow layer has
to stay sparse and keyed.
