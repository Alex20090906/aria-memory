"""H5 precondition (raised in the Gemini review): are the k-WTA units used evenly, or do a few "hub" units
win most of the time?  Per-token codes as the fast-weight memory computes them (layernorm, adaptation by
per-dimension z-score, random projection, top-k), over the tokens of a few thousand everyday sentences.

Reports, per layer and variant:
  dead       fraction of the D units never used
  top1%      share of all activations taken by the most used 1% of units
  gini       inequality of unit usage (0 = even)
  collision  measured mean overlap of two random token codes / the ideal k^2/D
Variants: z-score (current), ZCA whitening, and a BCM-like duty-cycle penalty (a unit that has been active
more than k/D of the time is penalised in later top-k choices).
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model1"))
import world3  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
D, K = 16384, 16


def gini(x):
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    H = model.config.hidden_size
    sents = []
    for x in world3.generate(20000, 3, "train"):
        if x["event"]:
            sents.append(x["event"].split("] ", 1)[1])
    rng = random.Random(0)
    rng.shuffle(sents)
    sents = sents[:1500]
    layers = [10, 14, 22, 25]
    X = {l: [] for l in layers}
    with torch.no_grad():
        for s in sents:
            out = model(tok(s, return_tensors="pt").input_ids.cuda(), output_hidden_states=True)
            for l in layers:
                X[l].append(nn.functional.layer_norm(out.hidden_states[l + 1][0, 1:].float(), (H,)))
    g = torch.Generator().manual_seed(0)
    P = (torch.randn(H, D, generator=g) / H ** 0.5).cuda()
    res = {}
    for l in layers:
        A = torch.cat(X[l])                         # tokens x H
        n = len(A)
        half = n // 2                              # statistics from the first half, codes measured on the second
        mu, sd = A[:half].mean(0), A[:half].std(0) + 1e-3
        Z = (A - mu) / sd
        C = torch.cov(((A[:half] - mu)).T)
        ev, V = torch.linalg.eigh(C)
        W = V @ torch.diag(1 / torch.sqrt(ev.clamp_min(1e-4))) @ V.T
        variants = {"zscore": Z[half:] @ P, "zca": ((A[half:] - mu) @ W) @ P}
        out = {}
        for name, S in variants.items():
            idx = S.topk(K, dim=1).indices.cpu().numpy()
            out[name] = idx
        # BCM-like duty cycle on the z-score scores: running activity a_j, penalty gamma * (a_j - k/D) / (k/D)
        S = (Z[half:] @ P)
        a = torch.full((D,), K / D, device=S.device)
        gamma, beta = float(S.std()) * 0.5, 0.01
        idx_b = []
        for t in range(len(S)):
            sc = S[t] - gamma * (a - K / D) / (K / D)
            top = sc.topk(K).indices
            idx_b.append(top.cpu().numpy())
            z = torch.zeros(D, device=S.device); z[top] = 1
            a = (1 - beta) * a + beta * z
        out["bcm"] = np.stack(idx_b)
        res[l] = {}
        for name, idx in out.items():
            cnt = np.bincount(idx.ravel(), minlength=D).astype(float)
            p = cnt / cnt.sum()
            srt = np.sort(cnt)[::-1]
            r_ = np.random.default_rng(0)
            a_, b_ = r_.integers(0, len(idx), 3000), r_.integers(0, len(idx), 3000)
            ov = np.mean([len(set(idx[i]) & set(idx[j])) for i, j in zip(a_, b_) if i != j])
            res[l][name] = {"tokens": int(len(idx)), "dead": round(float((cnt == 0).mean()), 3),
                            "top1%_share": round(float(srt[:D // 100].sum() / cnt.sum()), 3),
                            "gini": round(gini(cnt), 3), "collision_x_ideal": round(float(ov / (K * K / D)), 1)}
            print(l, name, res[l][name], flush=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results_hubness"), exist_ok=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "results_hubness", "hubness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
