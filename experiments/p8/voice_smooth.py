"""H39a: short-utterance voice smoothing (Gemini round 12, "relax the voice threshold for short utterances").

Almost every owner segment the voice check misses is short: under 1 s, 78 of 85 (old script) and 74 of 74 (new script)
are not labelled owner (many have no score at all). Rule: a segment shorter than DUR seconds that is not labelled
owner inherits "owner" when the nearest heard segment before or after it, within GAP seconds, is labelled owner by
its voice score. Counts on both script sets, against the timeline's true speakers (owner or the owner on the phone):
owner segments recovered, non-owner segments wrongly turned into owner, and told/never-told question segments that
become routable (some heard segment overlapping the question is owner).

    python voice_smooth.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.join(HERE, "..")
OWN = {"owner", "phone_owner"}
SETS = [("old", "llm_fastmem/asr.json", "llm_fastmem/timeline.json", "llm_fastmem/zh_scripts.json"),
        ("new", "realistic/set2/asr.json", "realistic/set2/audio/timeline.json", "realistic/set2/zh_scripts.json")]


def smooth(heard, dur, gap):
    out = []
    for i, h in enumerate(heard):
        lab = h["owner"]
        if lab != "owner" and h["end"] - h["start"] < dur:
            nb = []
            if i > 0 and h["start"] - heard[i - 1]["end"] <= gap:
                nb.append(heard[i - 1])
            if i + 1 < len(heard) and heard[i + 1]["start"] - h["end"] <= gap:
                nb.append(heard[i + 1])
            if any(n["owner"] == "owner" and n["end"] - n["start"] >= dur for n in nb):
                lab = "owner"
        out.append(lab)
    return out


def main():
    for dur, gap in ((1.0, 0.5), (2.0, 0.5), (2.0, 1.0), (3.0, 1.0)):
        tot = {"rec": 0, "false": 0, "q_told": 0, "q_told_new": 0, "q_never_new": 0}
        for name, ap, tp, sp in SETS:
            asr = json.load(open(os.path.join(X, ap), encoding="utf-8"))
            tl = {x["day"]: x for x in json.load(open(os.path.join(X, tp), encoding="utf-8"))}
            sc = json.load(open(os.path.join(X, sp), encoding="utf-8"))
            for d in asr:
                hs = d["heard"]
                new = smooth(hs, dur, gap)
                segs = tl[d["day"]]["segments"]
                for h, lab in zip(hs, new):
                    ov = [g for g in segs if min(h["end"], g["end"]) - max(h["start"], g["start"]) > 0.3]
                    if not ov or lab == h["owner"]:
                        continue
                    g = max(ov, key=lambda g: min(h["end"], g["end"]) - max(h["start"], g["start"]))
                    if g["speaker"] in OWN:
                        tot["rec"] += 1
                    else:
                        tot["false"] += 1
                for q in sc["questions"]:
                    if q["day"] != d["day"]:
                        continue
                    g = next((x for x in segs if x["text"] == q["text"]), None)
                    if g is None:
                        continue
                    idx = [i for i, h in enumerate(hs) if min(h["end"], g["end"]) - max(h["start"], g["start"]) > 0.3]
                    before = any(hs[i]["owner"] == "owner" for i in idx)
                    after = any(new[i] == "owner" for i in idx)
                    if q["expect"] is not None:
                        tot["q_told"] += 1
                        tot["q_told_new"] += (after and not before)
                    else:
                        tot["q_never_new"] += (after and not before)
        print(f"short < {dur}s, neighbour within {gap}s: owner segments recovered {tot['rec']} | non-owner turned "
              f"owner {tot['false']} | told question segments newly heard as owner {tot['q_told_new']}/{tot['q_told']} | "
              f"never-told newly heard {tot['q_never_new']}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
