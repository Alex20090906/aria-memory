"""Memory model v1: the engram moved inside the thinker (Qwen2.5-1.5B, frozen).

A fast-weight adapter sits on the residual stream after layer L:
    out = z(h) @ B,   added to h for every token
where B is the only thing that learns, online, while the model "experiences" each event
sentence (a few gradient steps on the sentence's own language-model loss). After each event
B decays toward 0 with tau_w (the user's decaying explicit weight w).

Two adapters are compared:
  sparse : z = k-winners-take-all of a fixed random expansion of LayerNorm(h) (pattern
           separation, like the dentate gyrus; same idea as engram.py but on the LLM's own
           hidden states)
  lora   : standard rank-r LoRA-style adapter, both factors learned (the usual test-time
           training baseline)

Familiarity (the user's slow threshold theta) lives on the same internal sparse code:
event code = k-WTA of the mean layer-L hidden state; theta keeps those codes with slow decay
tau_theta, familiarity = sum_i s_i (c . c_i)^2. Also reported: "surprise drop" = how much the
adapter lowers the sentence NLL, a familiarity signal some test-time-training work relies on.

Recall: from the partial event "On Tuesday, Axel put the passport", score all 10 places by
log-probability under the model + adapter; correct if the true place wins (chance 0.10).
"""
import argparse
import itertools
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from engram import OBJS, PLACES, DAYS, auc  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def sent(o, p, d):
    return f"{DAYS[d][3:]}, Axel put the {OBJS[o]} {PLACES[p]}."


def prefix(o, d):
    return f"{DAYS[d][3:]}, Axel put the {OBJS[o]}"


class SparseAdapter(nn.Module):
    def __init__(self, H, D, k, scale, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("P", torch.randn(H, D, generator=g) / H ** 0.5)
        self.B = nn.Parameter(torch.zeros(D, H))
        self.k, self.scale = k, scale
        self.ln = nn.LayerNorm(H, elementwise_affine=False)

    def code(self, h):
        a = self.ln(h.float()) @ self.P
        z = torch.zeros_like(a)
        z.scatter_(-1, a.topk(self.k, dim=-1).indices, 1 / self.k ** 0.5)
        return z

    def forward(self, h):
        return (self.code(h) @ self.B) * self.scale


class LoRAAdapter(nn.Module):
    def __init__(self, H, r, scale, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(H, r, generator=g) / H ** 0.5)
        self.B = nn.Parameter(torch.zeros(r, H))
        self.scale = scale
        self.ln = nn.LayerNorm(H, elementwise_affine=False)

    def forward(self, h):
        return (self.ln(h.float()) @ self.A @ self.B) * self.scale


def main(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    H = model.config.hidden_size
    layer = model.model.layers[args.layer]

    if args.kind == "sparse":
        ad = SparseAdapter(H, args.D, args.k, args.scale, args.seed).cuda()
    else:
        ad = LoRAAdapter(H, args.r, args.scale, args.seed).cuda()
    state = {"on": True}
    cap = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h
        if not state["on"]:
            return out
        h2 = h + ad(h).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    layer.register_forward_hook(hook)
    if args.code_layer >= 0 and args.code_layer != args.layer:
        def cap_hook(mod, inp, out):
            cap["hc"] = out[0] if isinstance(out, tuple) else out
        model.model.layers[args.code_layer].register_forward_hook(cap_hook)
    opt = torch.optim.Adam([ad.B], lr=args.lr) if args.kind == "sparse" else torch.optim.Adam(ad.parameters(), lr=args.lr)

    rng = random.Random(args.seed)
    combos = list(itertools.product(range(len(OBJS)), range(len(DAYS))))
    rng.shuffle(combos)
    combos = combos[:args.n]
    events = [(o, rng.randrange(len(PLACES)), d) for o, d in combos]
    seen_objs = {o for o, _ in combos}
    unseen = [o for o in range(len(OBJS)) if o not in seen_objs] or [0]
    sim_lures = [(o, (p + 1 + rng.randrange(len(PLACES) - 1)) % len(PLACES), d) for o, p, d in events]
    unrel = [(rng.choice(unseen), rng.randrange(len(PLACES)), rng.randrange(len(DAYS))) for _ in events]

    def ids(text):
        return tok(text, return_tensors="pt").input_ids.cuda()

    def nll(text):
        x = ids(text)
        with torch.no_grad():
            lg = model(x).logits[0, :-1].float()
        return float(nn.functional.cross_entropy(lg, x[0, 1:]))

    # pad side right; score each place continuation by summed log-prob of its tokens
    def recall(o, d):
        pre = prefix(o, d)
        n_pre = ids(pre).shape[1]
        texts = [f"{pre} {pl}." for pl in PLACES]
        enc = tok(texts, return_tensors="pt", padding=True)
        x, m = enc.input_ids.cuda(), enc.attention_mask.cuda()
        with torch.no_grad():
            lp = model(x, attention_mask=m).logits[:, :-1].float().log_softmax(-1)
        tgt = x[:, 1:]
        tl = lp.gather(-1, tgt[..., None])[..., 0]
        mask = m[:, 1:].clone()
        mask[:, :n_pre - 1] = 0
        return int((tl * mask).sum(1).argmax())

    gP = torch.Generator().manual_seed(999)
    Pf = (torch.randn(H, args.Df, generator=gP) / H ** 0.5).cuda()
    adapt = {"mu": None, "var": None, "n": 0}

    def event_code(text, learn=False):
        state_on = state["on"]
        state["on"] = False
        with torch.no_grad():
            model(ids(text))
        state["on"] = state_on
        # skip token 0 (attention-sink token with huge norm) and normalize each token first
        hs = nn.functional.layer_norm(cap.get("hc", cap["h"])[0, 1:].float(), (H,))
        # "last": final token state, which at a middle layer carries the whole sentence in context
        h = hs[-1] if args.code_from == "last" else hs.mean(0)
        if args.center:
            # sensory adaptation: subtract what all recent experiences have in common
            # (running mean/var, updated only while experiencing, frozen at probe time)
            if learn:
                adapt["n"] += 1
                if adapt["mu"] is None:
                    adapt["mu"], adapt["var"] = h.clone(), torch.ones_like(h)
                else:
                    # running average that settles (1/n), with a floor so it can still track slow drift
                    a_ = max(1 / adapt["n"], args.adapt_rate)
                    adapt["mu"] = (1 - a_) * adapt["mu"] + a_ * h
                    adapt["var"] = (1 - a_) * adapt["var"] + a_ * (h - adapt["mu"]) ** 2
            h = (h - adapt["mu"]) / (adapt["var"].sqrt() + 1e-3)
        a = h @ Pf
        z = torch.zeros_like(a)
        z[a.topk(args.kf).indices] = 1 / args.kf ** 0.5
        return z

    base_nll = {}
    state["on"] = False
    for e in events + sim_lures + unrel:
        base_nll[e] = nll(sent(*e))
    base_recall = [recall(o, d) == p for o, p, d in events]
    state["on"] = True

    codes, strength = [], []
    tw, tth = math.exp(-1 / args.tau_w), math.exp(-1 / args.tau_theta)
    for i, e in enumerate(events):
        x = ids(sent(*e))
        for _ in range(args.steps):
            lg = model(x).logits[0, :-1].float()
            loss = nn.functional.cross_entropy(lg, x[0, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ad.B.mul_(tw)
        strength = [s * tth for s in strength]
        codes.append(event_code(sent(*e), learn=True))
        strength.append(1.0)
        if i % 50 == 0:
            print(f"event {i} loss {loss.item():.3f}", flush=True)

    C = torch.stack(codes)
    S = torch.tensor(strength, device=C.device)

    def theta_fam(e):
        c = event_code(sent(*e))
        return float(((C @ c) ** 2 * S).sum())

    rows = []
    rec = np.array([recall(o, d) == p for o, p, d in events])
    drop = lambda e: base_nll[e] - nll(sent(*e))
    f_old = np.array([theta_fam(e) for e in events])
    f_sim = np.array([theta_fam(e) for e in sim_lures])
    f_unr = np.array([theta_fam(e) for e in unrel])
    d_old = np.array([drop(e) for e in events])
    d_sim = np.array([drop(e) for e in sim_lures])
    d_unr = np.array([drop(e) for e in unrel])
    ages = np.array([len(events) - 1 - i for i in range(len(events))])
    for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200)]:
        s = (ages >= lo) & (ages < hi)
        if s.sum() < 3:
            continue
        rows.append({"age": f"{lo}-{hi}", "n": int(s.sum()),
                     "recall": float(rec[s].mean()),
                     "recall_no_memory": float(np.mean(np.array(base_recall)[s])),
                     "theta_auc_similar": auc(f_old[s], f_sim[s]), "theta_auc_unrelated": auc(f_old[s], f_unr[s]),
                     "surprise_auc_similar": auc(d_old[s], d_sim[s]), "surprise_auc_unrelated": auc(d_old[s], d_unr[s])})
    # does the adapter damage unrelated language? NLL on generic text
    generic = ["The weather was nice, so we went for a walk in the park.",
               "Photosynthesis converts light energy into chemical energy.",
               "She opened the book and began to read the first chapter."]
    state["on"] = False
    g0 = np.mean([nll(t) for t in generic])
    state["on"] = True
    g1 = np.mean([nll(t) for t in generic])
    out = {"args": vars(args), "rows": rows, "generic_nll_base": g0, "generic_nll_with_memory": g1}
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    tag = f"inllm_{args.kind}_L{args.layer}_s{args.seed}{args.tag}"
    json.dump(out, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="sparse", choices=["sparse", "lora"])
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--D", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--tau_w", type=float, default=40.0)
    p.add_argument("--tau_theta", type=float, default=2000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--Df", type=int, default=32768)
    p.add_argument("--kf", type=int, default=20)
    p.add_argument("--center", type=int, default=1)
    p.add_argument("--adapt_rate", type=float, default=0.005)
    p.add_argument("--code_from", default="last", choices=["last", "mean"])
    p.add_argument("--code_layer", type=int, default=-1, help="layer for the familiarity code; -1 = same as adapter")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_inllm"))
    main(p.parse_args())
