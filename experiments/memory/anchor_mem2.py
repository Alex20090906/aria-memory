"""Anchor memory v2 (local model of LLM server head mode): which KEY survives a natural question before the anchor,
and which ANSWER key stops multi-word answers from drifting into other facts.  Design from Gemini round 4.

Fact: "[Leo|passport] -> in the red teapot."   Value: added at the head input (final norm output), exact
u_y - E_p[u] direction with logit gain dl*(1-p_y).  Memory never touches intermediate layers, so keys computed
from hidden states are the same with and without memory.

Anchor key (position of "->"):
  deep      k-WTA of the adapted state of "->" at layer Ld (current LLM server design)
  deep_sub  same, of h("->") - h("[")  (remove the context that "[" also carries)
  bilinear  k-WTA of (P x_subj) * roll(P x_obj, 1) at a shallow layer Ls, x = adapted mean state of the subject /
            object tokens (binds owner AND object; shallow so the question before it barely matters)
Answer keys (positions after "->", step m = 1, 2, ...):
  local     k-WTA of the token's own deep state (current LLM server design)
  shift     (anchor units + hash(m, token)) mod D: the anchor identity carried at full strength, permuted per step
  mix       k-WTA of roll(s_anchor, 1021 m) + 0.25 s_t (anchor carried, own state still counts)
Recall: C0 clean "[Leo|passport] ->", C1 "<natural question> [Leo|passport] ->"; greedy, must start with the place.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NAMES = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Axel", "Ken", "Lena", "Ivan", "Rosa", "Pablo", "Hana"]
OBJS = ["passport", "spare key", "watch", "glasses", "ticket", "charger", "ring", "wallet", "umbrella", "headphones",
        "notebook", "camera", "scarf", "medicine", "bank card", "laptop"]
PLACES = ["in the red teapot", "in the blue envelope", "in the garage drawer", "in the freezer", "in the toolbox",
          "in the rice jar", "in the old boot", "in the piano bench", "in the laundry basket", "in the glovebox",
          "under the doormat", "under the piano", "behind the bookshelf", "behind the mirror", "on the windowsill",
          "on top of the fridge"]
QFORMS = ["Where did {n} leave the {o}?", "Do you know where {n}'s {o} is?", "I can't find {n}'s {o}, any idea?",
          "Where is the {o} that {n} had?"]


def main(a):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    H = model.config.hidden_size
    D, k = a.D, a.k
    g = torch.Generator().manual_seed(a.seed)
    P = (torch.randn(H, D, generator=g) / H ** 0.5).cuda()
    B = torch.zeros(D, H, device="cuda")
    U = model.lm_head.weight.float()
    st = {"units": None, "off": False}
    stats = {}

    def head_hook(mod, inp, out):
        if st["off"] or st["units"] is None:
            return out
        T = out.shape[1]
        add = torch.zeros(T, H, device=out.device)
        for t, u in st["units"].items():
            if t < T:
                add[t] = B[u].sum(0) / k ** 0.5
        return out + add.to(out.dtype)[None]

    model.model.norm.register_forward_hook(head_hook)
    ids = lambda s: tok(s, return_tensors="pt").input_ids.cuda()
    pieces = lambda x: [tok.decode([int(i)]) for i in x[0]]

    def adapt(l, v):
        m, s = stats[l]
        return (v - m) / s

    def hstates(x):
        with torch.no_grad():
            st["off"] = True
            out = model(x, output_hidden_states=True)
            st["off"] = False
        return out.hidden_states

    def spans(x):
        p = pieces(x)
        a_end = max(i for i, s in enumerate(p) if "->" in s)
        lb = max(i for i in range(a_end) if "[" in p[i])
        bar = max(i for i in range(lb, a_end) if "|" in p[i])
        rb = max(i for i in range(bar, a_end + 1) if "]" in p[i])
        subj = list(range(lb + 1, bar)) or [lb]
        if "[" in p[lb] and len(p[lb].strip()) > 1 and not subj:
            subj = [lb]
        obj = list(range(bar + 1, rb)) or [bar]
        return a_end, lb, subj, obj

    def norm(v):
        return nn.functional.layer_norm(v.float(), (H,))

    def anchor_scores(hs, x):
        a_end, lb, subj, obj = spans(x)
        if a.key == "deep":
            s = adapt(a.Ld, norm(hs[a.Ld][0, a_end])) @ P
        elif a.key == "deep_sub":
            s = adapt("sub", norm(hs[a.Ld][0, a_end]) - norm(hs[a.Ld][0, lb])) @ P
        else:
            xs = adapt(("s", a.Ls), norm(hs[a.Ls][0, subj]).mean(0))
            xo = adapt(("o", a.Ls), norm(hs[a.Ls][0, obj]).mean(0))
            s = (xs @ P) * torch.roll(xo @ P, 1)
        return s, a_end

    def units_for(x):
        hs = hstates(x)
        s_anc, a_end = anchor_scores(hs, x)
        anc = s_anc.topk(k).indices
        out = {a_end: anc}
        toks = x[0].tolist()
        for t in range(a_end + 1, x.shape[1]):
            m = t - a_end
            if a.ans == "local":
                out[t] = (adapt(a.Ld, norm(hs[a.Ld][0, t])) @ P).topk(k).indices
            elif a.ans == "shift":
                h_ = (m * 1000003 + toks[t] * 7919) % D
                out[t] = (anc + h_) % D
            else:
                s_t = adapt(a.Ld, norm(hs[a.Ld][0, t])) @ P
                out[t] = (torch.roll(s_anc, 1021 * m) / (s_anc.std() + 1e-6) + 0.25 * s_t / (s_t.std() + 1e-6)).topk(k).indices
        return out

    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES for o in OBJS]
    rng.shuffle(pairs)
    facts = [(n, o, rng.choice(PLACES)) for n, o in pairs[:a.n]]
    anchor = lambda n, o: f"[{n}|{o}] ->"

    # adaptation statistics from other owner/object pairs (with and without a question before the anchor)
    acc = {}
    for i, (n, o) in enumerate(pairs[a.n:a.n + 120]):
        txt = f"{anchor(n, o)} {rng.choice(PLACES)}."
        if i % 2:
            txt = rng.choice(QFORMS).format(n=n, o=o) + "\n" + txt   # anchor starts a new line: same tokens as alone
        x = ids(txt)
        hs = hstates(x)
        a_end, lb, subj, obj = spans(x)
        acc.setdefault(a.Ld, []).append(norm(hs[a.Ld][0, 1:]))
        acc.setdefault("sub", []).append((norm(hs[a.Ld][0, a_end]) - norm(hs[a.Ld][0, lb]))[None])
        acc.setdefault(("s", a.Ls), []).append(norm(hs[a.Ls][0, subj]).mean(0)[None])
        acc.setdefault(("o", a.Ls), []).append(norm(hs[a.Ls][0, obj]).mean(0)[None])
    for key_, v in acc.items():
        V = torch.cat(v)
        stats[key_] = (V.mean(0), V.std(0) + 1e-3)

    @torch.no_grad()
    def write(text):
        x = ids(text)
        u = units_for(x)
        st["units"] = u
        lg = model(x).logits[0].float()
        p = lg[:-1].softmax(-1)
        for t, units in u.items():
            if t + 1 >= x.shape[1]:
                continue
            y = x[0, t + 1]
            du = U[y] - p[t] @ U
            B[units] += (a.dl * (1 - p[t, y]) * du / (du @ du + 1e-6)) / k ** 0.5

    @torch.no_grad()
    def recall(prompt, n_new=7):
        x = ids(prompt)
        n0 = x.shape[1]
        for _ in range(n_new):
            st["units"] = units_for(x)
            nxt = model(x).logits[0, -1].argmax()
            x = torch.cat([x, nxt.view(1, 1)], 1)
        return tok.decode(x[0, n0:])

    for n, o, p in facts:
        write(f"{anchor(n, o)} {p}.")
    c0, c1, ex = [], [], []
    for n, o, p in facts:
        r0 = recall(anchor(n, o)).strip()
        q = rng.choice(QFORMS).format(n=n, o=o)
        r1 = recall(f"{q}\n{anchor(n, o)}").strip()
        c0.append(r0.startswith(p)); c1.append(r1.startswith(p))
        if len(ex) < 4:
            ex.append([p, r0, r1])
    out = {"key": a.key, "ans": a.ans, "Ld": a.Ld, "Ls": a.Ls, "n": a.n, "C0": float(np.mean(c0)), "C1": float(np.mean(c1)),
           "examples": ex}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"v2_{a.key}_{a.ans}_Ls{a.Ls}_s{a.seed}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="deep", choices=["deep", "deep_sub", "bilinear"])
    p.add_argument("--ans", default="local", choices=["local", "shift", "mix"])
    p.add_argument("--Ld", type=int, default=18)
    p.add_argument("--Ls", type=int, default=2)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--D", type=int, default=16384)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--dl", type=float, default=14.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_anchor"))
    main(p.parse_args())
