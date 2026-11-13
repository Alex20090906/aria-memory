"""P8 offline, semantic finder: pick the label by Qwen3-Embedding similarity instead of character overlap.

Same replay and scoring as label_finder_eval.py. Policy: an exact label match is answered directly (as now);
otherwise the most similar earlier label is offered as an ask-back when its cosine similarity is >= tau.
Runs on CPU so the GPU stays free for the simulator.

    venv python label_finder_embed.py <zh_result_*.json> [...]
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_finder_eval as lf  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
INSTR = "Instruct: 給一句使用者的問話，找出它在問記憶裡的哪一件事（東西加上哪一方面）\nQuery: "


def run(path, enc, taus=(0.5, 0.6, 0.7, 0.8)):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(lf.FM, "timeline.json"), encoding="utf-8"))}
    log = [{"label": [w["fact"][0], w["fact"][1]], "value": w["fact"][2], "day": w["day"], "start": w["start"],
            "src": w["src"]} for w in d["writes"]]
    labels = sorted({tuple(ev["label"]) for ev in log})
    key_txt = [f"{e}（{SLOT_ZH.get(s, s)}）" for e, s in labels]
    K = enc.encode(key_txt, normalize_embeddings=True, batch_size=64)
    kidx = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for q in d["questions"]:
        qq = q["q"]
        seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        cand.sort(key=lambda a: -lf.overlap(qq["entity"], (a["anchor"] or "").split("|")[0]))
        a = cand[0] if cand else None
        told = qq["expect"] is not None
        if a is None or not a["anchor"]:
            rows.append({"told": told, "routed": False})
            continue
        e, s = (a["anchor"].strip("[]").split("|") + [""])[:2]
        before = [ev for ev in log if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                               and ev["start"] < seg["start"])]
        exact = lf.find("F0", e, s, qq["text"], before)
        latest = {}
        for ev in before:
            latest[tuple(ev["label"])] = ev["value"]
        best = None
        if latest:
            qv = enc.encode(INSTR + f"{qq['text']}（{e}，{SLOT_ZH.get(s, s)}）", normalize_embeddings=True)
            labs = list(latest)
            sims = K[[kidx[l] for l in labs]] @ qv
            j = int(np.argmax(sims))
            best = (labs[j], latest[labs[j]], float(sims[j]))
        rows.append({"told": told, "routed": True, "expect": qq["expect"], "entity": qq["entity"],
                     "anchor": f"[{e}|{s}]", "exact": exact, "best": best})
    print("==", os.path.basename(path))
    for tau in taus:
        c = dict(direct_ok=0, direct_wrong=0, ask_ok=0, ask_wrong=0, none=0, not_routed=0, never_direct=0, never_ask=0)
        for r in rows:
            if not r["routed"]:
                c["not_routed"] += r["told"]
                continue
            if r["exact"] is not None:
                if r["told"]:
                    c["direct_ok" if lf.overlap(r["expect"], r["exact"][0]) >= 0.6 else "direct_wrong"] += 1
                else:
                    c["never_direct"] += 1
                continue
            b = r["best"]
            if b is None or b[2] < tau:
                c["none"] += r["told"]
                continue
            if r["told"]:
                c["ask_ok" if lf.overlap(r["expect"], b[1]) >= 0.6 else "ask_wrong"] += 1
            else:
                c["never_ask"] += 1
        print(f"  tau {tau}: direct ok {c['direct_ok']} wrong {c['direct_wrong']} | ask-back ok {c['ask_ok']} wrong "
              f"{c['ask_wrong']} | none {c['none']} not routed {c['not_routed']} || never: direct {c['never_direct']}"
              f" ask-back {c['never_ask']}")
    for r in rows:
        if r["routed"] and r["exact"] is None and r["best"]:
            b = r["best"]
            ok = (lf.overlap(r["expect"], b[1]) >= 0.6) if r["told"] else None
            print(f"    {'told ' if r['told'] else 'never'} {r['entity']} {r['anchor']} -> {'|'.join(b[0])} "
                  f"{b[2]:.2f} {b[1][:20]!r} ok={ok}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    for p in sys.argv[1:]:
        run(p, enc)
