"""反射層（模型 1）在整條多天時間線上一直開著跑：每秒一個 tick，隱藏狀態從第 1 天一直保留到最後，從不重置。

用的是 H13 之後的反射層權重（model1/results_w6r/w6r_qwen3emb_fam_s0.pt：中文世界 v6、反射層標籤）。
輸入與 sensors/live.py 的 Model1Live 相同：[事件向量, 有沒有事件, 熟悉感, 思考模型忙不忙, 忙了多久]。
沒有事件的秒數一段一段批次送進 GRU（數學上與逐秒送相同），所以 21 天、181 萬個 tick 也能很快跑完。

簡化（寫在報告裡）：這一趟把「思考模型忙不忙」固定為 0，所以反射層的輸出與思考模型的狀態無關，可以先算好給所有對照組共用。
輸出 reflex_<tag>.json：每個事件的動作、要不要記、熟悉感；每小時安靜 tick 上的非 WAIT 次數。
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "model1"))
import train_gru3 as tg  # noqa: E402  (只讀取使用)


def ev_text(e):
    if e["kind"] == "cam":
        return f"[CAM] {e['text']}"
    who = "owner" if e["owner"] == "owner" else "someone"
    return f"[MIC] {who}: {e['text']}"


def main(tag="21d", ckpt=os.path.join(EXP, "model1", "results_w6r", "w6r_qwen3emb_fam_s0.pt")):
    from sentence_transformers import SentenceTransformer
    T = json.load(open(os.path.join(HERE, f"timeline_{tag}.json"), encoding="utf-8"))
    events = T["events"]
    enc = SentenceTransformer(tg.ENCODERS["qwen3emb"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    tg.EMB = enc.get_sentence_embedding_dimension()
    texts = sorted({ev_text(e) for e in events})
    vecs = dict(zip(texts, enc.encode(texts, normalize_embeddings=True, batch_size=128)))
    eng = tg.SparseEngram(seed=1)
    m = tg.Model1(tg.EMB + 5).cuda()
    m.load_state_dict(torch.load(ckpt))
    m.eval()
    A = tg.ACTIONS
    # 同一秒的事件合併成一個 tick（live.py 用 " | " 串起來）
    ticks = {}
    for e in events:
        ticks.setdefault(int(e["t"]), []).append(e)
    h, cur = None, 0
    per_event, quiet_false = {}, {}
    t0 = time.time()
    with torch.no_grad():
        for sec in sorted(ticks):
            gap = sec - cur
            while gap > 0:                                   # 安靜的秒數：批次送進去
                L = min(gap, 20000)
                x = torch.zeros(1, L, tg.EMB + 5, device="cuda")
                o, h = m(x, h)
                act = o[0, :, :5].argmax(-1)
                nz = torch.nonzero(act != 0).flatten().tolist()
                for j in nz:
                    hr = (cur + j) // 3600
                    quiet_false[hr] = quiet_false.get(hr, 0) + 1
                cur += L
                gap -= L
            evs = ticks[sec]
            text = " | ".join(ev_text(e) for e in evs)
            v = vecs[ev_text(evs[0])] if len(evs) == 1 else enc.encode(text, normalize_embeddings=True)
            x = np.zeros(tg.EMB + 5, np.float32)
            x[:tg.EMB] = v
            x[tg.EMB] = 1
            fam, fam_fast = eng.observe(torch.tensor(v, device="cuda", dtype=torch.float32), sec)
            x[tg.EMB + 1], x[tg.EMB + 4] = fam, fam_fast
            o, h = m(torch.tensor(x, device="cuda")[None, None], h)
            a = A[int(o[0, 0, :5].argmax())]
            g = bool(o[0, 0, 5] > 0)
            for e in evs:
                per_event[e["i"]] = {"action": a, "encode": g, "fam": round(float(fam), 3)}
            cur = sec + 1
    out = {"ckpt": os.path.basename(ckpt), "per_event": per_event, "quiet_false_per_hour": quiet_false,
           "ticks": cur, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(HERE, f"reflex_{tag}.json"), "w", encoding="utf-8"))
    from collections import Counter
    c = Counter((events[i]["src"] + ":" + (events[i]["truth"].get("speaker") or events[i]["truth"].get("type")), r["action"])
                for i, r in per_event.items())
    print("ticks", cur, "seconds", out["seconds"], "quiet false actions", sum(quiet_false.values()))
    print(json.dumps({f"{k}->{a}": n for (k, a), n in sorted(c.items())}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
