# Inline recall on LLM server (2026-11-04)

Setup: plain-text dialogue frame through the raw-prompt path (the raw path tokenizes special tokens as text, so no
chat template). Phase 1: thinker writes `[owner|object] ->` (stop at the arrow). Phase 2: same text continued with
the anchor set at the arrow row; the memory completes the place. Theta gate: anchor units' mean theta < 0.5 means
"never written" -> no injection, the thinker sees `(nothing)`. Phase 3: same text + "Aria's answer to the person:".
Engine anchor offset: 0 (the earlier -4 no longer applies on the current server build; verified by sweep).

Facts written: Leo passport -> in the freezer; Mia watch -> in the garage drawer. Each question twice.

| question | recalled place | final answer | ok |
|---|---|---|---|
| Where is Leo's passport? | in the freezer | "I remember it's in the freezer." (2/2) | 2/2 |
| Is Leo's passport likely to get frosted? (2-hop) | in the freezer | "Yes ... stored in the freezer, it will almost certainly get frost" / empty | 1/2 |
| Leo flies tomorrow: find it quickly, worry about condition? (2-hop) | in the freezer | reasons from the freezer both times (frost damage / retrieve carefully) | 2/2 |
| Mia's watch stopped: where to look, dusty? (2-hop) | in the garage drawer | "check the garage drawer; ... likely to be dusty" (2/2) | 2/2 |
| Where is Omar's wallet? (never written) | theta 0.0 -> (nothing) | "I do not remember where Omar's wallet is." (2/2) | 2/2 |

Recall 8/8, 2-hop reasoning 5/6 (one empty answer), theta gate 2/2 honest (before the gate: confabulated a place).

Capacity on LLM server (exp_c, dl 14, offset 0): n=40 seeds 0/1/2 C0 97.5/100/95, C1 100/95/100; n=100 C0 82 C1 90;
n=200 C0 55.5 C1 69.5. Local Qwen2.5-1.5B same keys: n=100 97/91, n=200 95/96.5.

## Update (same day, engine with token-located anchor, STRFANC2)

Engine commit 43aaa02: anchor.bin v2 carries the arrow token id; the engine locks the anchor at the first window
row carrying that token in the request (no absolute position; immune to prefix-cache state and draft counts).
Same 5 questions x 2: recall 10/10, 2-hop reasoning 8/8 (the empty answer is gone), theta gate 2/2 honest.

Earlier failure explained: with absolute positions the needed offset flipped between -4 and 0 between runs
(prefix cache / drafts); a 200-fact run with dl=22 scored 0 because the memory landed 3 tokens late.

## Capacity fixes (Gemini round 6 A1/A2), LLM server, dl 14, token-located anchor

| n | keys | write rule | C0 clean anchor | C1 natural question |
|---|---|---|---|---|
| 200 | bilinear | plain delta (base prior) | 55.5 | 69.5 |
| 200 | bilinear | A2 error-driven (prior read with memory on) | 91.0 | 91.5 |
| 200 | A1 cross-shifted tanh | plain delta | 74.5 | 93.0 |
| 200 | A1 cross-shifted tanh | A2 error-driven | **95.0** | **94.5** |
| 40 | A1 + A2 | | 97.5 | 100 |

A2 alone removes most of the majority-first-token effect (the write cancels what the shared units already say);
A1 adds the rest. Generic question unaffected (Paris).

## Sleep consolidation, first run (20 core + 180 noise, 200 events, sleep every 50, A1+A2 keys/writes)

| condition | core recall at end |
|---|---|
| fast decay tau 40, no sleep | 2/20 |
| never decay | 20/20 (with A1+A2 the 200-fact load no longer hurts; a harder test needs more noise) |
| sleep v1 (downscale everything not replayed this sleep) | 1/20 — bug: a fact consolidated at sleep 1 leaves the 64-event ring, is not replayed at sleep 2, and its units get downscaled x0.1. Fix: promoted (tau_slow) units are exempt from downscaling. |

Wake-time gated recalls during the run: 55/60 correct.

Sleep v2 (promoted units exempt from downscaling), 180 noise: 10/20. Failure analysis: the failed core facts were
written 30-41 events before a sleep; with tau_fast 40 the memory had decayed to 0.36-0.47 of dl by then and the
generative replay completed garbage ("[location|desk]"), which was then consolidated. Facts written within ~15 events
of a sleep (decay >= 0.65) all survived. v3: tau_fast 100 (worst case 0.61), unjudged events of the last wake window
protected from downscaling, 24x24 entity pool for the 480-noise test.

## Sleep v3 (tau_fast 100, unjudged window protected, promoted units exempt), 20 core facts

| noise | fast (tau 100, no sleep) | never decay | sleep |
|---|---|---|---|
| 180 (200 events) | 2/20 | 20/20 | **19/20** |
| 480 (500 events) | 2/20 | 16/20 | 15/20 |

At 480 noise the two oldest core facts fail in BOTH never and sleep by the theta gate (theta 0.45-0.49 < 0.5 after
~450 write events at theta tau 500), and three mid-run facts (written ~events 200-270) recall garbage in both.
v4: theta on promoted (tau_slow) units decays with tau 5000 (consolidation keeps familiarity too); replay values
are now logged against the truth to see whether the mid-run losses come from a wrong generative replay.

## Sleep v5 (ring stores names, replay plausibility gate), 480 noise, 20 core facts

| condition | core recall at end |
|---|---|
| fast (tau 100, no sleep) | 2/20 |
| never decay | 16/20 |
| sleep v5 | **17/20** |

v4 bug: the replay ring stored token ids, and "[Y"+"uki" tokenisation dropped the first letter of names, so replays
used wrong keys ("uki", "lex"). v5 stores the names. Replay gate ("reality monitoring"): a replay is consolidated
only if it looks like a place (preposition + noun phrase). 25 replays accepted, all 25 equal the truth; 3 rejected
("1000000", "2019-05", "2025-12"), those 3 facts were already corrupted at sleep time and are the 3 failures.
With 180 noise: sleep 19/20 vs never 20/20 vs fast 2/20.

## Colloquial test (40 Gemini-written casual facts, 60 chatter lines), LLM server, A1+A2

| version | gist | chatter false writes | casual | terse | "did I tell you" | someone else asks | honest on never-told |
|---|---|---|---|---|---|---|---|
| v1 "Name: text" frame | 27/30 | 0/60 | 13% | 30% | - | - | - |
| v2 'Name says: "..."' + few-shot anchors | 27/30 | 0/60 | 77% | 73% | 63% | 40% | 30-100% |
| v3 anchor resolved by a query-gist call | 27/30 | 0/60 | 80% | 73% | 83% | 80% | 100% all four |
| v3, writes computed in the thinker frame | 26/30 | 0/60 | 73% | 70% | 73% | 70% | 100% (worse: reverted) |

v1: "Name: text" made the thinker end its turn at once (no anchor). v2: free anchor writing picks the asker or copies
an example name ("[Leo|scarf]") for third-person asks. v3 misses: 2 facts were never written (gist failed) and are
correctly answered "I don't remember"; the rest are weak values in the chat frame (first token from memory, rest from
the prior: "in the rice jar" -> "in the top drawer of his desk"); the same memory in the "Memory notes." frame recalls
24/27. Source monitoring (memory on vs off) does not catch these: off mostly ends the turn.

## Gemini round 7 fixes (engine 762c5e5: relative anchor, answer keys without previous token)

| config | exp_c n=200 C0 | C1 | colloquial casual | terse | meta | other |
|---|---|---|---|---|---|---|
| A1+A2 (before) | 95.0 | 94.5 | 80 | 73 | 83 | 80 |
| + no-prev-token keys | | | 80 | 83 | 83 | 80 |
| + no-prev-token + error-driven at every step + margin dl | 77.0 | 92.0 | 80 | 80 | 87 | 83 |

Colloquial ceiling is 90% (3 of 30 facts were never written: gist failed). Remaining colloquial errors are strongly
counter-prior places ("notebook in the rice jar" -> "in the top drawer"); margin writes (up to 24 nats) still lose.
Error-driven-at-every-step hurts the clean-anchor recall at n=200 (digits appear: "in the 1.2.3"); being isolated.

Isolation (n=200 exp_c):
| config | C0 | C1 |
|---|---|---|
| A1+A2, prev-token keys (before) | 95.0 | 94.5 |
| A1+A2, no-prev-token keys | 92.0 | **97.0** |
| + margin dl (8-24 nats) | 67.0 | 73.5 |
| + margin + error-driven every step | 77.0 | 92.0 |
Colloquial with no-prev-token + margin (no every-step error-driven): 80/83/80/80 — margin buys nothing there.
Decision: default = A1 cross keys + A2 error-driven at the anchor step + no-prev-token answer keys. Margin and
every-step error-driven are off (bigger writes = more interference at n=200). Counter-prior places
("notebook in the rice jar") remain the main colloquial error.
