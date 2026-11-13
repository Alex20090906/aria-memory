"""Where the told questions of a retrieve-then-read run (zh_v3 with Z_READ=1, E42) are lost.

For every told question that was not answered right:
- not_routed: no answer was produced for the question's segment;
- not_heard: no logged fact available before the question carries the expected value (listening loss);
- not_retrieved: such a fact exists, but it is not among the k nearest rows to the heard question (retrieval loss);
- retrieved_said_unknown: it is among the k nearest, and the thinker said it does not know (reading loss);
- retrieved_wrong: it is among the k nearest, and the thinker answered with something else (reading loss).
Also counts how many of the matching facts exist only as night-relisten rows, and only under an English label.

The retrieval rank is recomputed with the same embedding model and instruction as the run (Qwen3-Embedding-0.6B);
staged rows are approximated by their logged write.

    python read_loss.py <zh_result_*.json> <timeline.json> [<zh_result_*.json> <timeline.json> ...]
"""
import json
import os
import re
import sys
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from label_finder_eval import overlap  # noqa: E402
from rescore_norm import norm, UNKNOWN  # noqa: E402

K = int(os.environ.get("RR_K", "5"))
SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
READ_INSTR = "Instruct: 給一句使用者的問話，找出記憶裡在講同一件事的紀錄\nQuery: "


def row_text(ev):
    e, s = ev["label"]
    return f"{e}｜{SLOT_ZH.get(s, s)}｜{ev['value']}（原話：{ev['heard']}）"


def english(e):
    return not re.search(r"[一-鿿]", e)


def run(path, tl_path, enc):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(tl_path, encoding="utf-8"))}
    log = [{"label": (w["fact"][0], w["fact"][1]), "value": str(w["fact"][2]), "day": w["day"], "start": w["start"],
            "src": w["src"], "heard": w.get("heard") or ""} for w in d["writes"]]
    R = enc.encode([row_text(ev) for ev in log], normalize_embeddings=True, batch_size=32)
    cats, detail = Counter(), []
    only_sleep = only_en = 0
    for r in d["questions"]:
        q = r["q"]
        if q["expect"] is None:
            continue
        ans = (r.get("answer") or "") + " " + str(r.get("tail") or "")
        if overlap(norm(q["expect"]), norm(ans)) >= 0.6 and not UNKNOWN.search(r.get("answer") or ""):
            cats["ok"] += 1
            continue
        seg = next((g for g in tl[q["day"]]["segments"] if g["text"] == q["text"]), None)
        if not r["routed"] or seg is None:
            cats["not_routed"] += 1
            continue
        avail = [i for i, ev in enumerate(log) if ev["day"] < q["day"] or
                 (ev["day"] == q["day"] and ev["src"] != "sleep" and ev["start"] < seg["start"])]
        match = [i for i in avail if overlap(norm(q["expect"]), norm(log[i]["value"])) >= 0.6]
        if not match:
            cats["not_heard"] += 1
            detail.append(("not_heard", q["day"], q["entity"], q["slot"], q["expect"]))
            continue
        if all(log[i]["src"] == "sleep" for i in match):
            only_sleep += 1
        if all(english(log[i]["label"][0]) for i in match):
            only_en += 1
        cand = [a for a in d["answers"] if a["day"] == q["day"] and
                min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        a = max(cand, key=lambda a: overlap(q["entity"], (a["anchor"] or "").split("|")[0])) if cand else None
        heard_q = (a or {}).get("heard") or q["text"]
        qv = enc.encode(READ_INSTR + heard_q, normalize_embeddings=True)
        sims = R[avail] @ qv
        order = [avail[j] for j in np.argsort(-sims)]
        rank = min(order.index(i) for i in match) + 1
        if rank > K:
            c = "not_retrieved"
        elif UNKNOWN.search(r.get("answer") or ""):
            c = "retrieved_said_unknown"
        else:
            c = "retrieved_wrong"
        cats[c] += 1
        best = min(match, key=lambda i: order.index(i))
        detail.append((c, q["day"], heard_q[:30], "rank", rank, row_text(log[best])[:60], "|ans:", (r.get("answer") or "")[:40]))
    told = sum(1 for r in d["questions"] if r["q"]["expect"] is not None)
    print("==", os.path.basename(path), f"told {told}", dict(cats),
          f"| fact only in night relisten: {only_sleep} | only under English label: {only_en}", flush=True)
    for x in detail:
        print("   ", x)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    args = sys.argv[1:]
    for p, t in zip(args[0::2], args[1::2]):
        run(p, t, enc)
