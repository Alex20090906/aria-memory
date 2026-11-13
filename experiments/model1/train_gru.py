"""Train a dedicated model 1: a small recurrent network that is always on.

Each tick it reads [event embedding, has_event, familiarity, thinker busy, busy time]
and outputs one of 5 actions. Its hidden state persists across all ticks forever: no
context window, no reset. Familiarity comes from the sparse engram (memory model v0),
so this run also tests the model 2 -> model 1 link.

Ablation --no_fam removes the familiarity input: then the network itself must notice
that a scene repeats within 400 s.
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

from world import ACTIONS, generate

EMB = 384


class SparseEngram:
    """Memory model v0 (see experiments/memory/engram.py): familiarity on a k-WTA code."""

    def __init__(self, D=32768, k=20, tau=2000.0, tau_fast=300.0, seed=0, device="cuda"):
        g = torch.Generator().manual_seed(seed)
        self.P = (torch.randn(EMB, D, generator=g) / EMB ** 0.5).to(device)
        self.k, self.tau, self.tau_fast, self.device = k, tau, tau_fast, device
        # bounded ring buffer of traces: traces older than the buffer are already faded
        self.cap = 3000
        self.codes = torch.zeros(self.cap, D, device=device, dtype=torch.float16)
        self.born = torch.full((self.cap,), -1e9, device=device)
        self.i = 0

    def observe(self, v, t):
        h = v @ self.P
        c = torch.zeros_like(h)
        c[h.topk(self.k).indices] = 1 / self.k ** 0.5
        overlap = (self.codes.float() @ c) ** 2
        # two components of the user's synapse: slow engram ("seen before") and
        # fast explicit trace ("seen just now")
        fam = float((overlap * torch.exp(-(t - self.born) / self.tau)).max())
        fam_fast = float((overlap * torch.exp(-(t - self.born) / self.tau_fast)).max())
        self.codes[self.i % self.cap] = c.half()
        self.born[self.i % self.cap] = float(t)
        self.i += 1
        return fam, fam_fast


def featurize(ticks, enc_cache, encoder, seed):
    texts = sorted({x["event"] for x in ticks if x["event"]} - set(enc_cache))
    if texts:
        vecs = encoder.encode(texts, normalize_embeddings=True, batch_size=256)
        enc_cache.update(zip(texts, vecs))
    eng = SparseEngram(seed=seed)
    X = np.zeros((len(ticks), EMB + 5), dtype=np.float32)
    for i, x in enumerate(ticks):
        if x["event"]:
            v = enc_cache[x["event"]]
            X[i, :EMB] = v
            X[i, EMB] = 1
            X[i, EMB + 1], X[i, EMB + 4] = eng.observe(torch.tensor(v, device="cuda"), x["t"])
        X[i, EMB + 2] = float(x["busy"])
        X[i, EMB + 3] = x["busy_for"] / 25
    y = np.array([ACTIONS.index(x["ideal"]) for x in ticks])
    return X, y


class Model1(nn.Module):
    def __init__(self, d_in, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, h), nn.GELU())
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.out = nn.Linear(h, len(ACTIONS))

    def forward(self, x, hstate=None):
        z, hstate = self.rnn(self.inp(x), hstate)
        return self.out(z), hstate


def evaluate(model, X, y, ticks, mask_fam, chunk=1000):
    model.eval()
    Xt = torch.tensor(X, device="cuda")
    if mask_fam:
        Xt[:, EMB + 1] = 0
        Xt[:, EMB + 4] = 0
    preds, h = [], None
    t0 = time.time()
    with torch.no_grad():
        # stream in chunks with a carried hidden state: identical to running tick by tick
        for i in range(0, len(Xt), chunk):
            o, h = model(Xt[None, i:i + chunk], h)
            preds.append(o[0].argmax(-1).cpu())
    p = torch.cat(preds).numpy()
    by = defaultdict(list)
    for x, pi in zip(ticks, p):
        by[x["kind"]].append(ACTIONS[pi] == x["ideal"])
    quiet = [ACTIONS[pi] != "WAIT" for x, pi in zip(ticks, p) if x["ideal"] == "WAIT"]
    need = [(i, ACTIONS[pi] == x["ideal"]) for i, (x, pi) in enumerate(zip(ticks, p)) if x["ideal"] != "WAIT"]
    q = len(ticks) // 4
    run_len, best = 0, 0
    for pi in p:
        run_len = run_len + 1 if pi != 0 else 0
        best = max(best, run_len)
    return {
        "per_kind": {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())},
        "n_kind": {k: len(v) for k, v in sorted(by.items())},
        "needs_action_acc": round(float(np.mean([c for _, c in need])), 3),
        "false_action_quiet": round(float(np.mean(quiet)), 4),
        "first_quarter": round(float(np.mean([c for i, c in need if i < q])), 3),
        "last_quarter": round(float(np.mean([c for i, c in need if i >= 3 * q])), 3),
        "longest_nonwait_run": int(best),
        "actions": dict(Counter(ACTIONS[i] for i in p)),
        "confusions": [[f"{k}->{a}", n] for (k, a), n in Counter(
            (x["kind"], ACTIONS[pi]) for x, pi in zip(ticks, p) if ACTIONS[pi] != x["ideal"]).most_common(8)],
        "ms_per_tick_batched": round((time.time() - t0) * 1000 / len(p), 4),
    }


def main(args):
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(args.seed)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    cache = {}
    t0 = time.time()
    train_ticks = generate(args.train_ticks, seed=args.seed, part="train")
    test_ticks = generate(args.test_ticks, seed=args.seed + 999, part="test")
    Xtr, ytr = featurize(train_ticks, cache, enc, args.seed)
    Xte, yte = featurize(test_ticks, cache, enc, args.seed + 1)
    print(f"features ready in {time.time() - t0:.0f}s; train {len(ytr)} test {len(yte)}", flush=True)

    model = Model1(Xtr.shape[1]).cuda()
    counts = np.bincount(ytr, minlength=len(ACTIONS)) + 1
    w = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32, device="cuda")
    lossf = nn.CrossEntropyLoss(weight=w / w.mean())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr, device="cuda"), torch.tensor(ytr, device="cuda")
    if args.no_fam:
        Xt[:, EMB + 1] = 0
        Xt[:, EMB + 4] = 0
    L, B = 512, 32
    for step in range(args.steps):
        model.train()
        starts = torch.randint(0, len(Xt) - L, (B,))
        xb = torch.stack([Xt[s:s + L] for s in starts])
        yb = torch.stack([yt[s:s + L] for s in starts])
        o, _ = model(xb)
        loss = lossf(o.reshape(-1, len(ACTIONS)), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)

    res = evaluate(model, Xte, yte, test_ticks, args.no_fam)
    # true always-on latency: one tick at a time with persistent hidden state
    model.eval()
    h, x1 = None, torch.tensor(Xte[:1000], device="cuda")
    with torch.no_grad():
        t1 = time.time()
        for i in range(len(x1)):
            _, h = model(x1[None, i:i + 1], h)
        torch.cuda.synchronize()
    res["ms_per_tick_online"] = round((time.time() - t1) * 1000 / len(x1), 3)
    res["params"] = sum(p.numel() for p in model.parameters())
    res["config"] = vars(args)
    print(json.dumps(res, indent=1))
    os.makedirs(args.out, exist_ok=True)
    name = f"gru2_{'nofam' if args.no_fam else 'fam'}_s{args.seed}"
    json.dump(res, open(os.path.join(args.out, name + ".json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(args.out, name + ".pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_ticks", type=int, default=200000)
    p.add_argument("--test_ticks", type=int, default=30000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_fam", action="store_true")
    p.add_argument("--out", default="results_gru")
    main(p.parse_args())
