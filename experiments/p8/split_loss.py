"""Split the told-question losses of a zh_v3 run into: fact never written (listening loss), fact written under another
label (finding loss), and written under the right label but still not answered right (recall loss).

    python split_loss.py <zh_result_*.json> [...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rescore_norm import norm, UNKNOWN  # noqa: E402
from label_finder_eval import overlap  # noqa: E402


def split(path):
    d = json.load(open(path, encoding="utf-8"))
    wrote = {}
    for w in d["writes"]:
        if w.get("fact"):
            wrote.setdefault((w["fact"][0], w["fact"][1]), []).append(str(w["fact"][2]))
    vals = [v for vs in wrote.values() for v in vs]
    out = {"ok": 0, "not_heard": 0, "other_label": 0, "same_label_wrong": 0}
    for r in d["questions"]:
        q = r["q"]
        if q["expect"] is None:
            continue
        a = (r.get("answer") or "") + " " + str(r.get("tail") or "")
        if overlap(norm(q["expect"]), norm(a)) >= 0.6 and not UNKNOWN.search(r.get("answer") or ""):
            out["ok"] += 1
        elif any(overlap(norm(q["expect"]), norm(v)) >= 0.6 for v in wrote.get((q["entity"], q["slot"]), [])):
            out["same_label_wrong"] += 1
        elif any(overlap(norm(q["expect"]), norm(v)) >= 0.6 for v in vals):
            out["other_label"] += 1
        else:
            out["not_heard"] += 1
    by_kind = {}
    for r in d["questions"]:
        q = r["q"]
        if q["expect"] is None:
            continue
        k = by_kind.setdefault(q["kind"], [0, 0])
        k[1] += 1
        a = (r.get("answer") or "") + " " + str(r.get("tail") or "")
        k[0] += overlap(norm(q["expect"]), norm(a)) >= 0.6 and not UNKNOWN.search(r.get("answer") or "")
    out["by_kind"] = {k: f"{a}/{b}" for k, (a, b) in by_kind.items()}
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        print(os.path.basename(p), json.dumps(split(p), ensure_ascii=False))
