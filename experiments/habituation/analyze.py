"""Score always-on runs: loop metrics, event responsiveness, fluency.

Usage: python analyze.py results [--nll]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

WIN = 250          # self tokens per timeline window
RESP_WIN = 150     # self tokens allowed to answer an event


def load(folder):
    runs = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        if os.path.basename(p).startswith("summary"):
            continue
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def distinct4(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / max(len(grams), 1)


def segments(tok, records):
    """Split the stream into sentences, each tagged with the self-token index where it ends."""
    sents, buf, n_self, has_event = [], [], 0, False
    for tid, src, _ in records:
        buf.append(tid)
        if src == "self":
            n_self += 1
        else:
            has_event = True
        piece = tok.decode([tid])
        if re.search(r"[.!?\n]", piece):
            text = tok.decode(buf).strip()
            if len(text.split()) >= 4 and not has_event:
                sents.append((n_self, text))
            buf, has_event = [], False
    return sents


def event_scores(tok, run):
    recs = run["records"]
    out = []
    for ev in run["events"]:
        start = ev["at_record"]
        # skip the injected tokens, then take RESP_WIN self tokens
        i = start
        while i < len(recs) and recs[i][1] == "event":
            i += 1
        resp, latency = [], None
        for j in range(i, min(i + RESP_WIN, len(recs))):
            if recs[j][1] != "self":
                break
            resp.append(recs[j][0])
            if latency is None:
                text = tok.decode(resp).lower()
                if any(k in text for k in ev["keys"]):
                    latency = len(resp)
        text = tok.decode(resp)
        ok = latency is not None and "say" in text.lower()
        out.append({"q": ev["text"], "ok": ok, "latency": latency, "reply": text[:200]})
    return out


def analyze(folder, do_nll=False):
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    runs = load(folder)
    tok = AutoTokenizer.from_pretrained(runs[0]["model"])
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    nll_model = None
    if do_nll:
        import torch
        from transformers import AutoModelForCausalLM
        nll_model = AutoModelForCausalLM.from_pretrained(runs[0]["model"], dtype=torch.float16, device_map="cuda")

    per_run = []
    for run in runs:
        self_ids = [t for t, s, _ in run["records"] if s == "self"]
        n = len(self_ids)
        d4_t = [distinct4(self_ids[i:i + WIN]) for i in range(0, n, WIN)]

        sents = segments(tok, run["records"])
        sem_t = [[] for _ in range((n + WIN - 1) // WIN)]
        dup = 0
        if len(sents) > 1:
            vecs = emb.encode([s for _, s in sents], normalize_embeddings=True)
            seen = set()
            for k in range(len(sents)):
                if sents[k][1] in seen:
                    dup += 1
                seen.add(sents[k][1])
                if k == 0:
                    continue
                prev = vecs[max(0, k - 30):k]
                sim = float((prev @ vecs[k]).max())
                w = min((sents[k][0] - 1) // WIN, len(sem_t) - 1)
                sem_t[w].append(sim)
        sem_t = [float(np.mean(x)) if x else float("nan") for x in sem_t]

        ents = [e for _, s, e in run["records"] if s == "self"]
        evs = event_scores(tok, run)

        nll = None
        if nll_model is not None:
            import torch
            ids = tok(run["prompt"], return_tensors="pt").input_ids[0].tolist()
            plen = len(ids)
            ids = ids + [t for t, _, _ in run["records"]]
            mask = [False] * plen + [s == "self" for _, s, _ in run["records"]]
            with torch.no_grad():
                x = torch.tensor([ids], device="cuda")
                logits = nll_model(x).logits[0, :-1].float()
                lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0]
                m = torch.tensor(mask[1:], device="cuda")
                nll = float(-lp[m].mean())

        self_text = tok.decode(self_ids)
        fake_sensor = len(re.findall(r"\[(MIC|CAM)\]|Person:", self_text)) * 1000 / max(n, 1)

        half = len(d4_t) // 2
        per_run.append({
            "name": run["name"], "seed": run["seed"],
            "distinct4_late": float(np.nanmean(d4_t[half:])),
            "semsim_late": float(np.nanmean(sem_t[half:])),
            "dup_sent_frac": dup / max(len(sents), 1),
            "n_sents": len(sents),
            "entropy_late": float(np.mean(ents[n // 2:])),
            "events_ok": sum(e["ok"] for e in evs), "events_n": len(evs),
            "latency_mean": float(np.mean([e["latency"] for e in evs if e["latency"] is not None]))
            if any(e["latency"] is not None for e in evs) else None,
            "nll_self": nll, "fake_sensor_per_1k": fake_sensor,
            "distinct4_t": d4_t, "semsim_t": sem_t, "events": evs,
            "tail": tok.decode(self_ids[-300:]),
        })

    groups = defaultdict(list)
    for r in per_run:
        groups[r["name"]].append(r)

    def ms(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            return "-"
        return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"

    cols = ["distinct4_late", "semsim_late", "dup_sent_frac", "fake_sensor_per_1k", "nll_self"]
    print(f"{'condition':<18}{'n':>3}  " + "  ".join(f"{c:>16}" for c in cols) + f"  {'events':>8}  {'latency':>8}")
    summary = {}
    for name, rs in sorted(groups.items()):
        ok = sum(r["events_ok"] for r in rs)
        tot = sum(r["events_n"] for r in rs)
        lat = ms([r["latency_mean"] for r in rs])
        print(f"{name:<18}{len(rs):>3}  " + "  ".join(f"{ms([r[c] for r in rs]):>16}" for c in cols)
              + f"  {ok:>3}/{tot:<4}  {lat:>8}")
        summary[name] = {c: [r[c] for r in rs] for c in cols}
        summary[name]["events_ok"] = ok
        summary[name]["events_n"] = tot

    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": summary, "runs": per_run}, f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, rs in sorted(groups.items()):
        sem = np.nanmean(np.array([r["semsim_t"] for r in rs], dtype=float), axis=0)
        d4 = np.nanmean(np.array([r["distinct4_t"] for r in rs], dtype=float), axis=0)
        x = (np.arange(len(sem)) + 0.5) * WIN
        ax[0].plot(x, sem, marker="o", label=name)
        ax[1].plot(x, d4, marker="o", label=name)
    for a, t in zip(ax, ["semantic self-similarity (higher = looping)", "distinct 4-grams (lower = looping)"]):
        a.set_title(t)
        a.set_xlabel("self tokens")
        for pos in (500, 900, 1300, 1700, 2100):
            a.axvline(pos, color="gray", lw=0.5, ls="--")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "timeline.png"), dpi=120)
    print("wrote summary.json and timeline.png")


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results", "--nll" in sys.argv)
