"""Rescore always-on runs (told 51 / never-told 21) with numbers and weekdays normalised (as p8/rescore_norm.py) and
one stricter rule: when the expected answer names a day number (第N天) and the answer names a different day number,
it is wrong. The character-overlap check alone counted "第9天早上9點" as right for "第21天早上九點".

    python rescore_strict.py <run tag> [...]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p8"))
from rescore_norm import norm  # noqa: E402
import analyze as A  # noqa: E402

DAY = re.compile(r"第\s*(\d+)\s*天")


def right(expect, tail):
    if A.overlap(norm(expect), norm(tail)) < 0.6:
        return False
    de, dt = DAY.findall(expect), DAY.findall(tail)
    return not (de and dt and de[0] != dt[0])


def score(tag):
    r = json.load(open(os.path.join(HERE, "results", f"run_Dd_21d_s0_{tag}.json"), encoding="utf-8"))
    hit = wrong = hon = nev = 0
    for q in A.SC["questions"]:
        g = next((x for x in A.TL[q["day"]]["segments"] if x["text"] == q["text"]), None)
        cand = [x for x in r["log"]["answers"] if g and x["day"] == q["day"] and x["hstart"] is not None
                and min(x["hend"], g["end"]) - max(x["hstart"], g["start"]) > 0.3]
        cand.sort(key=lambda x: -A.overlap(q["entity"], x["anchor"].split("|")[0]))
        a = cand[0] if cand else None
        if q["expect"] is None:
            nev += 1
            hon += a is None or not a["passed"]
        else:
            ok = bool(a and a["passed"] and right(q["expect"], a["tail"]))
            hit += ok
            wrong += bool(a and a["passed"] and not ok)
    print(f"{tag}: right {hit}/51, answered wrong {wrong}, never-told kept {hon}/{nev}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for t in sys.argv[1:]:
        score(t)
