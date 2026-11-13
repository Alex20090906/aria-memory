"""P8 offline, write side (H27): when a new fact's label is very close to a label already in the log, write it under
the existing label instead of opening a new one (熔蠟鍋 / 融蠟鍋, 存折 / 存摺). Then lookup = exact label, else the
semantic finder with ask-back at tau 0.6 (as E37). Compares merge thresholds kappa on the replayed writes of a run.

    python write_merge_eval.py <zh_result_*.json> [...]        (P8_FM=<folder with timeline.json> for set2)
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


def key_text(l):
    return f"{l[0]}（{SLOT_ZH.get(l[1], l[1])}）"


def run(path, enc, kappas=(None, 0.95, 0.9, 0.85, 0.8), tau=0.6):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(lf.FM, "timeline.json"), encoding="utf-8"))}
    raw = [{"label": (w["fact"][0], w["fact"][1]), "value": w["fact"][2], "day": w["day"], "start": w["start"],
            "src": w["src"]} for w in d["writes"]]
    labs = sorted({ev["label"] for ev in raw})
    V = dict(zip(labs, enc.encode([key_text(l) for l in labs], normalize_embeddings=True, batch_size=64)))
    qrows = []
    for q in d["questions"]:
        qq = q["q"]
        seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        cand.sort(key=lambda a: -lf.overlap(qq["entity"], (a["anchor"] or "").split("|")[0]))
        a = cand[0] if cand else None
        if a is None or not a["anchor"] or seg is None:
            qrows.append(None)
            continue
        e, s = (a["anchor"].strip("[]").split("|") + [""])[:2]
        qv = enc.encode(INSTR + f"{qq['text']}（{e}，{SLOT_ZH.get(s, s)}）", normalize_embeddings=True)
        qrows.append((qq, seg, (e, s), qv))
    print("==", os.path.basename(path), flush=True)
    for kappa in kappas:
        log, merged = [], 0
        for ev in raw:
            lab = ev["label"]
            if kappa is not None and log and lab not in {x["label"] for x in log}:
                prev = list(dict.fromkeys(x["label"] for x in log if x["label"][1] == lab[1]))   # same slot only
                if prev:
                    sims = np.stack([V[p] for p in prev]) @ V[lab]
                    j = int(np.argmax(sims))
                    if sims[j] >= kappa:
                        lab = prev[j]
                        merged += 1
            log.append(dict(ev, label=lab))
        c = dict(direct_ok=0, direct_wrong=0, ask_ok=0, ask_wrong=0, none=0, never_direct=0, never_ask=0)
        for row in qrows:
            if row is None:
                continue
            qq, seg, (e, s), qv = row
            told = qq["expect"] is not None
            before = [ev for ev in log if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                                   and ev["start"] < seg["start"])]
            latest = {}
            for ev in before:
                latest[ev["label"]] = ev["value"]
            if (e, s) in latest:
                if told:
                    c["direct_ok" if lf.overlap(qq["expect"], latest[(e, s)]) >= 0.6 else "direct_wrong"] += 1
                else:
                    c["never_direct"] += 1
                continue
            ls = list(latest)
            if not ls:
                c["none"] += told
                continue
            sims = np.stack([V[l] for l in ls]) @ qv
            j = int(np.argmax(sims))
            if sims[j] < tau:
                c["none"] += told
            elif told:
                c["ask_ok" if lf.overlap(qq["expect"], latest[ls[j]]) >= 0.6 else "ask_wrong"] += 1
            else:
                c["never_ask"] += 1
        print(f"  merge {'off ' if kappa is None else kappa:4}: merged {merged:3d} writes | right {c['direct_ok'] + c['ask_ok']:2d} "
              f"(direct {c['direct_ok']}, ask {c['ask_ok']}) | direct wrong {c['direct_wrong']} | ask wrong {c['ask_wrong']} "
              f"| none {c['none']} | never: direct {c['never_direct']} ask {c['never_ask']}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    for p in sys.argv[1:]:
        run(p, enc)
