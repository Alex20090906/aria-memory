# 7. Negative results

Ideas that were tested and did not hold. Many were plausible and some came from design rounds with a strong model;
each row says what was measured. Ids refer to `experiments/map/研究地圖.md`, where every node lists its result.

| Id | Idea | What happened | Source |
|---|---|---|---|
| E1 | Make a generator always on by suppressing repetition (repetition penalty, attention-synapse depression) | repetition penalty: NLL 0.40 to 2.80, 2/9 questions answered; depression: repetition 20% to 80% | chapter 1.1; `experiments/habituation/results_a/` |
| C1 | A prompted general LLM as the always-on model | no fixation, but could not tell TV from the owner, ignored greetings, sent everything to the thinker | `experiments/阶段报告_01.md`, C1 |
| B2 | LoRA as online memory | recall 0.27 right after the event; general NLL 2.01 to 8.38 after 150 events | chapter 2.1 |
| H2 | Familiarity as a sentence-average code | AUC 0.68 (works only on anchor keys: 1.00) | map H2 |
| H12 | Let familiarity choose among attribute guesses | honesty on never-told questions 100% to 33% | chapter 2.6 |
| H14 | First-token probability as confidence | kills true memories: right 26 to 19 | chapter 2.5 |
| H17 | Error-correcting codes for stored answers | worse at high load; 3/750 collapsed at D = 4,096 | map H17 |
| H18 | Direction coherence of units as confidence | AUC 0.51–0.67 | map H18 |
| H19 | Block addressing extends retention 8 times | the occupied fraction does not change; the derivation treated an error rate as a concentration bound | map row H19 |
| H20 | Compile fast weights into dense slow weights with null-space protection | 20 facts harm general language, 100 break it | chapter 2.6 |
| — | "Automatically raise write strength" (Gemini round 7) | 200-fact recall fell to 67–73%; turned off | log entry 2026-11-05 |
| — | "The pseudo-inverse is needed for the write direction" | the transpose reached 0.97 | `experiments/階段報告_04.md` |
| E19, E27 | Freeze or softly protect consolidated rows | 29/20 and 30/19 right/wrong against 35/15 unprotected | chapter 3 |
| E29 | Sleep more often | 37 vs 35 of 60; oldest quarter still 1/15 | chapter 3 |
| H23 | Effective rank collapse as a vegetative-state marker | rank rose, 1,860 to 2,798, in the no-brake group | chapter 1.3 |
| E9 | The reflex model routes speech by meaning | 9–13 of 40 owner utterances answered on real ASR text | chapter 1.2 |
| H25 | Find labels by pinyin sound score | set2 17 to 18, set_old 28 to 28, one more wrong ask | map H25 |
| H27 | Merge similar labels at write time | set_old 28 to 26 (merged two different medicines) | map H27 |
| H31 | Tell the reader that sound-alike names count | never-told answered 0 to 3 | chapter 6.5 |
| H32b | A rule for the misheard "哪/拿" | chit-chat taken as a question twice as often | chapter 6.5 |
| H33 | Give the reader the previous sentences | no reliable gain (within 1 question) | chapter 6.5 |
| H34 | Answer first, cite the record after | not faster; 4/40 picks changed | map H34 |
| H37 | Ask back with the index's best guess | right in 4 of 16 | chapter 6.5 |
| H38 | Targeted unlearning of old versions (α = 0.6) | baseline 32/34 (worse: other aspects of the same object weakened); with 10,000 distractors 19/30 vs 18/29 | map H38; `experiments/p8/eval_h38.txt` |
| H38a, H40 in E48 | Split meaning/sound quotas, clean false facts end to end | 29/51 vs 31/51, within code-reassignment noise | chapter 4.4 |
| H39 | Short-utterance voice continuation, ASR n-best | +1/102; 67 to 68 right and 9 to 10 wrong | chapter 6.3 |
| — | Index readout strength as a "have I heard this" gate | AUC 0.72–0.83 | chapter 4.3 |
| — | Gemini's "the echo gate closes as rows are scaled" | the angle does not depend on the scale | map D10 |

## Patterns

- **Confidence signals cannot catch a true answer to the wrong question.** When a never-told question is answered,
  the answer is usually the real content of a similar name (H22), so its confidence is high.
- **Most "smart" additions on the reader side traded honesty for recall.** The rules that survived are narrow
  (exact pinyin match, strict same-thing-same-aspect reading).
- **Several failures were measurement artefacts first.** A timing bug doubled waits (chapter 5.2), a whole-second
  file timestamp hid updates (chapter 2.7), case-sensitive scoring moved results by one (chapter 3.2), an anchor
  offset made one 200-question run all wrong (chapter 2.7). Each was found by reading individual answers.
