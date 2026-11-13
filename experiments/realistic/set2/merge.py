"""Merge Gemini's three replies (days 1-7, 8-14, 15-21, written in one Antigravity conversation) into zh_scripts.json.

Same shape and checks as gen_zh_scripts.py. Nothing in the text is changed; only update_of indices are shifted to the
merged facts array. Questions whose text is not found verbatim among the owner's segments are reported (zh_v3 scores a
question by finding its segment in the timeline, so such a question would count as not routed).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    allb = {"days": [], "facts": [], "questions": []}
    for i in (1, 2, 3):
        b = json.load(open(os.path.join(HERE, f"reply_{i}.json"), encoding="utf-8"))
        off = len(allb["facts"])
        for f_ in b["facts"]:
            if f_.get("update_of") is not None:
                f_["update_of"] += off
        allb["days"] += b["days"]; allb["facts"] += b["facts"]; allb["questions"] += b["questions"]
    days = [d["day"] for d in allb["days"]]
    assert days == list(range(1, len(days) + 1)), days
    bad = [f_ for f_ in allb["facts"] if f_["slot"] not in SLOTS]
    badq = [q for q in allb["questions"] if q["slot"] not in SLOTS]
    seg_text = {(d["day"], s["text"]) for d in allb["days"] for s in d["segments"] if s["speaker"] == "owner"}
    missing = [q for q in allb["questions"] if (q["day"], q["text"]) not in seg_text]
    print("days", len(days), "facts", len(allb["facts"]), "questions", len(allb["questions"]),
          "told", sum(q["expect"] is not None for q in allb["questions"]),
          "segments", sum(len(d["segments"]) for d in allb["days"]), "bad slots", len(bad), len(badq),
          "questions not in segments", len(missing))
    for q in missing:
        print("  missing:", q["day"], q["text"])
    json.dump(allb, open(os.path.join(HERE, "zh_scripts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
