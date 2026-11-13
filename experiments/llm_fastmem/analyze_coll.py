"""Failure breakdown for colloquial_result_v2.json."""
import collections
import json
import os
import sys

p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                                        "colloquial_result_v2.json")
d = json.load(open(p, encoding="utf-8"))
print(json.dumps(d["summary"]))
gist_bad = {w["line"].split(":")[0] + "|" + str(w["gist"]) for w in d["writes"] if not w.get("chatter") and not w.get("ok")}
c = collections.Counter()
ex = collections.defaultdict(list)
for a in d["asks"]:
    if a["told"] and not a["hit"]:
        if not a["anchor"]:
            why = "no anchor written"
        elif a["tail"] == "(nothing)":
            why = "theta gate closed (key mismatch?)"
        elif a.get("idk"):
            why = "recalled but answered don't know"
        else:
            why = "recalled wrong / answer lost"
        c[(a["kind"], why)] += 1
        ex[why].append((a["kind"], a["anchor"], (a["tail"] or "")[:25], a["answer"][:70]))
    if not a["told"] and not a["idk"]:
        c[(a["kind"], "NEVER-TOLD: not honest")] += 1
        ex["never told, not honest"].append((a["kind"], a["anchor"], (a["tail"] or "")[:25], a["answer"][:70]))
for k, v in sorted(c.items()):
    print(k, v)
for why, rows in ex.items():
    print("==", why)
    for r in rows[:6]:
        print("  ", r)
