"""Rescore GPUtw sleep.py result files with case-insensitive matching (the instance's sleep.py predates the fix).

For each file: told questions (correct / honest forget / confident wrong), the same after a weakest-token confidence
threshold applied afterwards, recall by age quartile (young -> old), never-told questions answered, and how many
correct answers the model also guesses with the memory switched off (prior_guess).

    python rescore_gputw.py results_gputw/*.json
"""
import json
import os
import sys


def rescore(path, taus=(0.5, 0.7)):
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    for r in rows:
        p = r["fact"].split(": ", 1)[1]
        out = r["out"]
        r["ok"] = out.lower().startswith(p.lower())
        r["honest_forget"] = (not r["ok"]) and out.strip() == ""
        r["confident_wrong"] = (not r["ok"]) and out.strip() != ""
        r["prior_ok"] = bool(r.get("prior_guess")) and r["prior_guess"].lower().startswith(p.lower())
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    hf = sum(r["honest_forget"] for r in rows)
    cw = sum(r["confident_wrong"] for r in rows)
    by_age = sorted(rows, key=lambda r: r["age"])
    q = [by_age[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    quart = [f"{sum(r['ok'] for r in g)}/{len(g)}" for g in q]
    out = {"file": os.path.basename(path), "mode": d.get("mode"), "n": n, "correct": ok, "honest_forget": hf,
           "confident_wrong": cw, "quartiles_young_to_old": quart,
           "prior_ok_of_correct": sum(r["ok"] and r["prior_ok"] for r in rows)}
    for t in taus:
        keep = [r for r in rows if r["pmin"] is not None and r["pmin"] >= t]
        out[f"pmin>={t} correct/wrong"] = f"{sum(r['ok'] for r in keep)}/{sum(not r['ok'] and r['out'].strip() != '' for r in keep)}"
    nv = d.get("never") or []
    if nv:
        ans = [r for r in nv if r["answered"]]
        out["never_answered"] = f"{len(ans)}/{len(nv)}"
        out["never_answered_pmin>=0.7"] = sum(1 for r in ans if r["pmin"] is not None and r["pmin"] >= 0.7)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        print(json.dumps(rescore(p), ensure_ascii=False))
