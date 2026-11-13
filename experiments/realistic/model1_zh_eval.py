"""Model 1 on what it would really hear: ASR utterances of the rendered Chinese days, one tick per second.

Input event per heard utterance: "[MIC] owner: <asr text>" when owner-voice says owner, else "[MIC] someone: <text>"
(the w5 style; w4 used names). The checkpoints were trained on the ENGLISH simulated worlds: this measures zero-shot
transfer. Ideal labels come from the true segments under each utterance:
  owner -> aria, a question in the script          THINK, write
  owner -> aria, otherwise                          REPLY, write
  owner -> family/visitor/phone (overheard)         WAIT,  write
  family / visitor / tv                             WAIT,  no write   (single user: others are background)
Reports action accuracy and write-gate accuracy per kind, and the false action rate on silent ticks.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = os.path.join(HERE, "..", "model1")
sys.path.insert(0, M1)
import train_gru3 as tg  # noqa: E402


def build_ticks():
    asr = json.load(open(os.path.join(HERE, os.environ.get("ASR_FILE", "asr.json")), encoding="utf-8"))
    tl = {d["day"]: d for d in json.load(open(os.path.join(HERE, "audio", "timeline.json"), encoding="utf-8"))}
    sc = json.load(open(os.path.join(HERE, "zh_scripts.json"), encoding="utf-8"))
    qtexts = {(q["day"], q["text"]) for q in sc["questions"]}
    ticks, kinds = [], []
    T = 0
    for day in asr:
        last = 0
        for h in day["heard"]:
            t_end = int(h["end"]) + 1
            for _ in range(max(0, t_end - last - 1)):          # silent seconds
                ticks.append({"t": T, "event": None, "busy": 0, "busy_for": 0, "ideal": "WAIT", "encode": 0}); kinds.append("silence"); T += 1
            last = t_end
            segs = [g for g in tl[day["day"]]["segments"] if min(g["end"], h["end"]) - max(g["start"], h["start"]) > 0.3]
            main = max(segs, key=lambda g: min(g["end"], h["end"]) - max(g["start"], h["start"])) if segs else None
            who = "owner" if h["owner"] == "owner" else "someone"
            ev = f"[MIC] {who}: {h['text']}"
            if main is None:
                ideal, enc, kind = "WAIT", 0, "unmatched"
            elif main["speaker"] in ("owner", "phone_owner"):
                if main["to"] == "aria":
                    if (day["day"], main["text"]) in qtexts:
                        ideal, enc, kind = "THINK", 1, "owner_question"
                    else:
                        ideal, enc, kind = "REPLY", 1, "owner_to_aria"
                else:
                    ideal, enc, kind = "WAIT", 1, "owner_overheard"
            else:
                ideal, enc, kind = "WAIT", 0, main["speaker"]
            if os.environ.get("REFLEX") == "1":      # reflex layer: owner speech -> ATTEND (hand to listen), others WAIT
                ideal, enc = ("ATTEND", 1) if kind.startswith("owner") else ("WAIT", 0)
            ticks.append({"t": T, "event": ev, "busy": 0, "busy_for": 0, "ideal": ideal, "encode": enc}); kinds.append(kind); T += 1
        for _ in range(30):
            ticks.append({"t": T, "event": None, "busy": 0, "busy_for": 0, "ideal": "WAIT", "encode": 0}); kinds.append("silence"); T += 1
    return ticks, kinds


def main():
    from sentence_transformers import SentenceTransformer
    ticks, kinds = build_ticks()
    enc = SentenceTransformer(tg.ENCODERS["qwen3emb"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    tg.EMB = enc.get_sentence_embedding_dimension()
    rows = {}
    for ck in sys.argv[1:] or ["w5_qwen3emb_fam_s0.pt", "w4_qwen3emb_fam_s0.pt"]:
        sub = "results_" + ck.split("_")[0]
        X, (y, ye) = tg.featurize(ticks, {}, enc, seed=1)
        m = tg.Model1(X.shape[1]).cuda()
        m.load_state_dict(torch.load(os.path.join(M1, sub, ck)))
        m.eval()
        with torch.no_grad():
            o, _ = m(torch.tensor(X, device="cuda")[None])
        act = o[0, :, :5].argmax(-1).cpu().numpy()
        gate = (o[0, :, 5] > 0).cpu().numpy()
        per = defaultdict(lambda: [0, 0, 0])
        for k_, a, yy, g, e in zip(kinds, act, y, gate, ye):
            per[k_][0] += int(a == yy); per[k_][1] += int(g == bool(e)); per[k_][2] += 1
        rows[ck] = {k_: f"act {v[0]}/{v[2]}  gate {v[1]}/{v[2]}" for k_, v in sorted(per.items())}
        print(ck, json.dumps(rows[ck], ensure_ascii=False, indent=1), flush=True)
    json.dump(rows, open(os.path.join(HERE, "model1_zh_eval.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
