"""H35 at scale: how many logged rows can the network index hold before finding breaks down?

Distractor rows are recombinations of real rows from both script sets: the entity and aspect of one row, the value of a
second and the heard sentence of a third. They share the real vocabulary, its word frequencies and its sounds, so they
are hard negatives (many share the asked entity or value words), not random noise. N distractors are written first
(as older days), then the real log of one script set is replayed as in index_eval.py, and recall@5 of the told
questions whose answer is in the log is measured for the network index (LLM server word vectors + pinyin).

    python index_scale.py <zh_result_*.json> <timeline.json> <other zh_result for distractor parts> [...]
env: IX_DOUT (512), IX_NS ("0,2000,10000,50000")
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("IX_EMB", "llm")
import numpy as np  # noqa: E402
import index_eval as ix  # noqa: E402
from label_finder_eval import overlap  # noqa: E402
from rescore_norm import norm  # noqa: E402


def merge(*cnts):
    out = {}
    for c in cnts:
        for u, n in c.items():
            out[u] = out.get(u, 0) + n
    return out


def run(path, tl_path, part_paths, keys, ns):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(tl_path, encoding="utf-8"))}
    log = [{"label": (w["fact"][0], w["fact"][1]), "value": str(w["fact"][2]), "day": w["day"], "start": w["start"],
            "src": w["src"], "heard": w.get("heard") or ""} for w in d["writes"]]
    DENSE = os.environ.get("IX_DENSE") == "1"
    parts = []
    for p in [path] + part_paths:
        for w in json.load(open(p, encoding="utf-8"))["writes"]:
            parts.append((f"{w['fact'][0]} {ix.SLOT_ZH.get(w['fact'][1], w['fact'][1])}", str(w["fact"][2]),
                          w.get("heard") or ""))
    t0 = time.time()
    raw_e = [keys.raw(a) for a, _, _ in parts]
    raw_v = [keys.raw(b) for _, b, _ in parts]
    raw_h = [keys.raw(c) for _, _, c in parts]
    print(f"   parts {len(parts)} keyed in {time.time() - t0:.0f}s", flush=True)
    raws = [keys.raw(f"{ev['label'][0]} {ix.SLOT_ZH.get(ev['label'][1], ev['label'][1])} {ev['value']} {ev['heard']}")
            for ev in log]
    qs = []
    for r in d["questions"]:
        q = r["q"]
        if q["expect"] is None:
            continue
        seg = next((g for g in tl[q["day"]]["segments"] if g["text"] == q["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == q["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        if cand:
            a = max(cand, key=lambda a: overlap(q["entity"], (a["anchor"] or "").split("|")[0]))
            qs.append((q, seg, a.get("heard") or q["text"], keys.raw(a.get("heard") or q["text"])))
    for n_d in ns:
        rng = np.random.default_rng(int(os.environ.get("IX_SEED", "11")))   # codes
        drng = np.random.default_rng(11)                                     # distractors: the same set for every code scheme
        W = np.zeros((ix.D, ix.DOUT), np.float32)
        n_use = np.zeros(ix.D, np.float32)
        C = (rng.choice(np.array([-1.0, 1.0], np.float32), (n_d + len(log), ix.DOUT)) / np.sqrt(ix.DOUT)).astype(np.float32)
        codes = os.environ.get("IX_CODES", "random")
        if codes in ("ortho", "block"):
            # H41b: codes beyond DOUT rows. ortho = the first DOUT rows orthogonal, the rest random; block = every block
            # of DOUT consecutive rows is orthogonal within itself (blocks independent of each other)
            nb = 1 if codes == "ortho" else (len(C) + ix.DOUT - 1) // ix.DOUT
            for b in range(nb):
                q_, _ = np.linalg.qr(rng.standard_normal((ix.DOUT, ix.DOUT)))
                k = min(ix.DOUT, len(C) - b * ix.DOUT)
                C[b * ix.DOUT:b * ix.DOUT + k] = q_[:k].astype(np.float32)

        keys_store, wlist = {}, []

        def write(cnt, row):
            ix.write_row(W, C, n_use, keys_store, wlist, row, cnt)

        t0 = time.time()
        dtexts = []
        for j in range(n_d):
            a, b, c = drng.integers(0, len(parts), 3)
            write(merge(raw_e[a], raw_v[b], raw_h[c]), j)
            dtexts.append(f"{parts[a][0]}｜{parts[b][1]}（原話：{parts[c][2]}）")
        if DENSE:
            from read_loss import READ_INSTR, row_text
            RD = ENC.encode(dtexts, normalize_embeddings=True, batch_size=64) if dtexts else np.zeros((0, 1024), np.float32)
            RR = ENC.encode([row_text(ev) for ev in log], normalize_embeddings=True, batch_size=64)
            dense_hit = 0
        t_d = time.time() - t0
        order = sorted(range(len(log)), key=lambda i: (log[i]["day"], log[i]["src"] == "sleep", log[i]["start"]))
        events = [("w", (log[i]["day"], 1 if log[i]["src"] == "sleep" else 0, log[i]["start"]), i) for i in order]
        events += [("q", (q["day"], 0, seg["start"]), j) for j, (q, seg, _, _) in enumerate(qs)]
        events.sort(key=lambda x: (x[1], 0 if x[0] == "w" else 1))
        written, hit, n = list(range(n_d)), 0, 0
        for kind, _, j in events:
            if kind == "w":
                write(raws[j], n_d + j)
                written.append(n_d + j)
                continue
            q, seg, heard, qraw = qs[j]
            real = [r - n_d for r in written if r >= n_d]
            avail = [i for i in real if log[i]["day"] < q["day"] or (log[i]["src"] != "sleep" and log[i]["start"] < seg["start"])]
            match = {n_d + i for i in avail if overlap(norm(q["expect"]), norm(log[i]["value"])) >= 0.6}
            if not match:
                continue
            n += 1
            u, w = ix.sparse(qraw, n_use)
            y = w @ W[u]
            cand = list(range(n_d)) + [n_d + i for i in avail]
            s = C[cand] @ y
            top = {cand[k] for k in np.argpartition(-s, 5)[:5]}
            hit += bool(top & match)
            if DENSE:
                qv = ENC.encode(READ_INSTR + heard, normalize_embeddings=True)
                sc = np.concatenate([RD @ qv, RR[avail] @ qv])
                dtop = {cand[k] for k in np.argpartition(-sc, 5)[:5]}
                dense_hit += bool(dtop & match)
        print(f"  {os.path.basename(path)} D={ix.DOUT} distractors {n_d}: recall@5 {hit}/{n}"
              + (f" | text embedding {dense_hit}/{n}" if DENSE else "") + f"  (distractor writes {t_d:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    ENC = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    keys = ix.Keys(ENC)
    ns = [int(x) for x in os.environ.get("IX_NS", "0,2000,10000,50000").split(",")]
    a = sys.argv[1:]
    run(a[0], a[1], a[2:], keys, ns)
