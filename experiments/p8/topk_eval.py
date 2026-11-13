"""P8 offline: is the right label among the top-k semantic candidates? If top-3 holds far more right answers than
top-1, letting the thinker choose among a few candidates (or asking back with two) is worth an end-to-end run.

    python topk_eval.py <zh_result_*.json> [...]        (P8_FM=<folder with timeline.json> for set2)
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_finder_eval as lf  # noqa: E402
import numpy as np  # noqa: E402
from label_finder_embed import INSTR, SLOT_ZH  # noqa: E402


def run(path, enc, ks=(1, 2, 3, 5)):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(lf.FM, "timeline.json"), encoding="utf-8"))}
    log = [{"label": [w["fact"][0], w["fact"][1]], "value": w["fact"][2], "day": w["day"], "start": w["start"],
            "src": w["src"]} for w in d["writes"]]
    labels = sorted({tuple(ev["label"]) for ev in log})
    K = enc.encode([f"{e}（{SLOT_ZH.get(s, s)}）" for e, s in labels], normalize_embeddings=True, batch_size=64)
    kidx = {lab: i for i, lab in enumerate(labels)}
    hit = {k: 0 for k in ks}
    hit_tau = {k: 0 for k in ks}
    never_tau = {k: 0 for k in ks}
    n_inexact = n_reach = 0
    for q in d["questions"]:
        qq = q["q"]
        seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        cand.sort(key=lambda a: -lf.overlap(qq["entity"], (a["anchor"] or "").split("|")[0]))
        a = cand[0] if cand else None
        if a is None or not a["anchor"]:
            continue
        e, s = (a["anchor"].strip("[]").split("|") + [""])[:2]
        before = [ev for ev in log if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                               and ev["start"] < seg["start"])]
        if lf.find("F0", e, s, qq["text"], before) is not None:
            continue
        latest = {}
        for ev in before:
            latest[tuple(ev["label"])] = ev["value"]
        if not latest:
            continue
        labs = list(latest)
        qv = enc.encode(INSTR + f"{qq['text']}（{e}，{SLOT_ZH.get(s, s)}）", normalize_embeddings=True)
        sims = K[[kidx[l] for l in labs]] @ qv
        order = np.argsort(-sims)
        told = qq["expect"] is not None
        if told:
            n_inexact += 1
            ok = [lf.overlap(qq["expect"], latest[labs[j]]) >= 0.6 for j in order]
            n_reach += any(ok)
        for k in ks:
            top = order[:k]
            if told:
                hit[k] += any(ok[:k])
                hit_tau[k] += any(ok[i] and sims[order[i]] >= 0.6 for i in range(min(k, len(order))))
            else:
                never_tau[k] += bool(sims[top[0]] >= 0.6)
    print("==", os.path.basename(path), f"| told questions without an exact label: {n_inexact}, right value somewhere "
          f"in memory: {n_reach}")
    for k in ks:
        print(f"  top-{k}: right label among them {hit[k]} (with cosine >= 0.6: {hit_tau[k]}) | never-told with a "
              f"candidate >= 0.6: {never_tau[k]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    for p in sys.argv[1:]:
        run(p, enc)
