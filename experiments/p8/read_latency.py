"""H34: where the ~2.2 s of one retrieve-then-read answer goes (E44), and what a spoken reply would wait for.

Replays the told questions of a zh_v3 result (as retrieve_read_eval.py) against the normal LLM server service (8081),
one request at a time, and times:
  embed_q     the question embedding (Qwen3-Embedding-0.6B on this PC's CPU, as the embedding service does)
  embed_rows  embedding the rows logged since the previous question (E44 does this at question time)
  ttft        time to the first streamed token of the reader
  first_line  time until the reader's first line is complete ("編號 N" / "不知道": the answer is decided here)
  total       the whole reader output
for two output orders:
  pick_first  current: line 1 = 編號 N or 不知道, line 2 = the reply
  reply_first line 1 = the reply sentence, line 2 = 編號 N or 不知道 (speech could start at the first token, but the
              network gate on the picked fact is only known at the end)
Also reports whether both orders pick the same row.

    python read_latency.py <zh_result_*.json> [n_questions]       (P8_FM=<folder with timeline.json> for set2)
"""
import json
import os
import re
import sys
import time
import urllib.request

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_finder_eval as lf  # noqa: E402
import numpy as np  # noqa: E402
from retrieve_read_eval import INSTR, SYS, URL, row_text  # noqa: E402

SYS_REPLY_FIRST = SYS.replace("輸出兩行：第一行「編號 N」或「不知道」；第二行用一句話回答使用者（不知道就說不記得）。",
                              "輸出兩行：第一行用一句話回答使用者（不知道就說不記得）；第二行「編號 N」或「不知道」。")
assert SYS_REPLY_FIRST != SYS


# H34b: RL_PREFIX=1 pins a LLM server checkpoint at the end of the system message ("llm_prefix": {"messages": 1}), so
# each read prefills only the question and its 5 rows. Without it, a short prompt (< 2,048 tokens) gets no checkpoint
# at the end of its system prompt and every read is prefilled from the start (measured: same prompt again 0.24 s to the
# first token, same system prompt with other rows 1.27 s).
PREFIX = os.environ.get("RL_PREFIX") == "1"
ORDERS = os.environ.get("RL_ORDERS", "pick_first,reply_first").split(",")


def stream(system, user, max_tokens=80):
    req_body = {"model": "qwen3.8-flash-next-iq3_s", "stream": True, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "chat_template_kwargs": {"enable_thinking": False}}
    if PREFIX:
        req_body["llm_prefix"] = {"messages": 1}
    body = json.dumps(req_body, ensure_ascii=False).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    t0 = time.time()
    ttft = first_line = None
    text = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
            except Exception:
                continue
            if delta and ttft is None:
                ttft = time.time() - t0
            text += delta
            if first_line is None and "\n" in text.strip("\n"):
                first_line = time.time() - t0
    total = time.time() - t0
    return text, ttft or total, first_line or total, total


def pick_of(text, top, reply_first):
    lines = [x for x in text.strip().splitlines() if x.strip()]
    line = (lines[1] if len(lines) > 1 else "") if reply_first else (lines[0] if lines else "")
    m = re.search(r"編號\s*(\d+)", line)
    return top[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= len(top) else None


def main(path, n_max, enc):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(lf.FM, "timeline.json"), encoding="utf-8"))}
    log = [{"label": (w["fact"][0], w["fact"][1]), "value": str(w["fact"][2]), "day": w["day"], "start": w["start"],
            "src": w["src"], "heard": w.get("heard") or ""} for w in d["writes"]]
    vec, done = {}, 0
    rows = {"embed_q": [], "embed_rows": [], "n_new_rows": []}
    for order in ORDERS:
        rows.update({f"{order}_{k}": [] for k in ("ttft", "first_line", "total")})
    same = n = 0
    pick_log = []
    for q in d["questions"]:
        qq = q["q"]
        if qq["expect"] is None or n >= n_max:
            continue
        seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        if not cand:
            continue
        heard_q = cand[0].get("heard") or qq["text"]
        idx = [i for i, ev in enumerate(log) if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                                          and ev["start"] < seg["start"])]
        if not idx:
            continue
        new = [i for i in idx if i not in vec]
        t0 = time.time()
        if new:
            for i, v in zip(new, enc.encode([row_text(log[i]) for i in new], normalize_embeddings=True, batch_size=32)):
                vec[i] = v
        rows["embed_rows"].append(time.time() - t0)
        rows["n_new_rows"].append(len(new))
        t0 = time.time()
        qv = enc.encode(INSTR + heard_q, normalize_embeddings=True)
        rows["embed_q"].append(time.time() - t0)
        sims = np.stack([vec[i] for i in idx]) @ qv
        top = sorted([idx[j] for j in np.argsort(-sims)[:5]], key=lambda i: (log[i]["day"], log[i]["start"]))
        lines = [f"{k + 1}. 第{log[i]['day']}天｜{row_text(log[i])}" for k, i in enumerate(top)]
        user = f"今天是第{qq['day']}天。\n使用者問：{heard_q}\n記憶紀錄：\n" + "\n".join(lines)
        picks = {}
        for order, system in [x for x in (("pick_first", SYS), ("reply_first", SYS_REPLY_FIRST)) if x[0] in ORDERS]:
            text, ttft, first_line, total = stream(system, user)
            rows[f"{order}_ttft"].append(ttft)
            rows[f"{order}_first_line"].append(first_line)
            rows[f"{order}_total"].append(total)
            picks[order] = pick_of(text, top, order == "reply_first")
        same += len(set(map(str, picks.values()))) == 1
        pick_log.append(picks.get("pick_first"))
        n += 1
    print("==", os.path.basename(path), f"questions {n}", flush=True)
    for k, v in rows.items():
        v = np.array(v, float)
        print(f"  {k:24s} median {np.median(v):.3f}  p90 {np.percentile(v, 90):.3f}  max {v.max():.3f}", flush=True)
    print(f"  same pick in both orders: {same}/{n}", flush=True)
    print("  picks", json.dumps(pick_log), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40, enc)
