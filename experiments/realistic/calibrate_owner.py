"""Owner-voice threshold from the first week only (an enrolment week), applied to every day; reports days 8-21.
Rule: just above the highest non-owner score seen in week 1 (others becoming "me" is the costly error)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(a, tl, days):
    out = []
    for d in a:
        if d["day"] not in days:
            continue
        for h in d["heard"]:
            segs = [g for g in tl[d["day"]]["segments"] if min(g["end"], h["end"]) - max(g["start"], h["start"]) > 0.3]
            if not segs or h["score"] is None:
                continue
            main = max(segs, key=lambda g: min(g["end"], h["end"]) - max(g["start"], h["start"]))
            out.append((h["score"], main["speaker"] in ("owner", "phone_owner")))
    return out


def main():
    a = json.load(open(os.path.join(HERE, "asr.json"), encoding="utf-8"))
    tl = {d["day"]: d for d in json.load(open(os.path.join(HERE, "audio", "timeline.json"), encoding="utf-8"))}
    cal, test = rows(a, tl, set(range(1, 8))), rows(a, tl, set(range(8, 22)))
    th = max(s for s, o in cal if not o) + 0.01
    P = sum(o for _, o in test); N = len(test) - P
    tp = sum(1 for s, o in test if o and s >= th); fp = sum(1 for s, o in test if not o and s >= th)
    print(f"threshold {th:.3f}; days 8-21: owner {tp}/{P} ({tp / max(1, P):.0%}), others accepted {fp}/{N}")
    for d in a:
        for h in d["heard"]:
            if h["score"] is not None:
                h["owner"] = "owner" if h["score"] >= th else "other"
    json.dump(a, open(os.path.join(HERE, "asr.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
