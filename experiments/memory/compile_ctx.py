"""Context compilation: write fast weights WITHOUT backpropagation (forward passes only), so the
same rule can run inside an inference-only engine such as LLM server.

The user's "bidirectional neuron compiler": during a conversation an event lives in the context
(KV cache, episodic). Compiling it into weights means: make the model, without the context,
behave as if the context were there.

For each token position t of the fact sentence S, at each memory layer l:
    h_ctx(t)  = residual after layer l when S was already seen in context  ("S\nS", second copy)
    h_bare(t) = residual after layer l for S alone                        (adapters ON, so earlier
                memory layers' effect is included)
    target    = h_ctx(t) - h_bare(t)            what the context added at this point
    z         = kWTA(adapt(h_bare(t)))          sparse code of the bare state (pattern separation)
    B_l      += eta * z^T (target - z B_l)      delta rule: B_l learns to add the missing part

Recall needs no context: the cue prefix runs through the model with the adapters on.
Compared with gradient writing (inllm.py): recall, familiarity is unchanged (theta), generic NLL.
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
from engram import OBJS, PLACES, DAYS  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    H = model.config.hidden_size
    layers = [int(x) for x in a.layers.split(",")]
    g = torch.Generator().manual_seed(a.seed)
    P = {l: (torch.randn(H, a.D, generator=g) / H ** 0.5).cuda() for l in layers}
    B = {l: torch.zeros(a.D, H, device="cuda") for l in layers}
    stat = {l: {"mu": None, "var": None, "n": 0} for l in layers}
    st = {"on": True, "cap": {}}

    def norm(l, h):
        hn = nn.functional.layer_norm(h.float(), (H,))
        s = stat[l]
        if s["mu"] is not None:
            hn = (hn - s["mu"]) / (s["var"].sqrt() + 1e-3)
        return hn

    def code(l, h):
        a_ = norm(l, h) @ P[l]
        z = torch.zeros_like(a_)
        z.scatter_(-1, a_.topk(a.k, dim=-1).indices, 1 / a.k ** 0.5)
        return z

    def mk(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            st["cap"][l] = h
            st.setdefault("pre", {})[l] = h
            if not st["on"]:
                return out
            h2 = h + (code(l, h) @ B[l]).to(h.dtype)
            dl = st.get("delta", {}).get(l)
            if dl is not None:                      # probe / gradient handle on this layer's residual
                h2 = h2 + dl.to(h2.dtype)
            st["cap"][l] = h2
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    for l in layers:
        model.model.layers[l].register_forward_hook(mk(l))

    ids = lambda t: tok(t, return_tensors="pt").input_ids.cuda()

    @torch.no_grad()
    def adapt(text):
        on, st["on"] = st["on"], False
        model(ids(text))
        st["on"] = on
        for l in layers:
            hs = nn.functional.layer_norm(st["cap"][l][0, 1:].float(), (H,))
            s = stat[l]
            s["n"] += 1
            r = max(1 / s["n"], 0.005)
            tm, tv = hs.mean(0), hs.var(0, unbiased=False)
            if s["mu"] is None:
                s["mu"], s["var"] = tm.clone(), tv + 1e-2
            else:
                s["var"] = (1 - r) * s["var"] + r * (tv + (tm - s["mu"]) ** 2)
                s["mu"] = (1 - r) * s["mu"] + r * tm

    @torch.no_grad()
    def compile_fact(text):
        """Forward passes only. Returns nothing; updates B."""
        x = ids(text)
        n = x.shape[1]
        sep = ids("\n")
        both = torch.cat([x, sep, x], 1)
        for it in range(a.iters):
            st["on"] = False
            lc = model(both).logits[0, -n:].float()
            h_ctx = {l: st["cap"][l][0, -n:].float().clone() for l in layers}
            st["on"] = True
            lb = model(x).logits[0].float()          # bare, with current memory
            # only compile where the context still changes what the model would say next
            # (KL from context to bare prediction); shared, ambiguous positions get little weight
            kl = (lc.softmax(-1) * (lc.log_softmax(-1) - lb.log_softmax(-1))).sum(-1)
            wpos = (kl / (kl.max() + 1e-6)).clamp(0, 1)[:, None] if a.kl_weight else torch.ones(n, 1, device=x.device)
            if a.target == "logit":
                # output-direction target: for position t the next token y is known; the change of the final
                # residual that raises log p(y) is W_U[y] - E_p[W_U] (only the softmax and the unembedding,
                # both forward quantities: no backprop through the network)
                y = x[0, 1:]
                p_b = lb[:-1].softmax(-1)
                U = model.lm_head.weight.float()
                if a.no_prob:
                    # engine-friendly variant: needs only the unembedding rows, not the model's own prediction
                    d = U[y] - U.mean(0)
                    miss = torch.ones(n - 1, 1, device=x.device)
                else:
                    d = U[y] - p_b @ U
                    miss = (1 - p_b.gather(-1, y[:, None]))          # how much is still missing
                d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
                tgt = torch.zeros(n, H, device=x.device)
                tgt[:-1] = a.gain * miss * d
            if a.target in ("grad", "spsa"):
                # H4: the direction that raises log p(next token) AT THIS LAYER (the model's own local geometry),
                # not the unembedding direction.  "grad": exact (one backward pass, the oracle);
                # "spsa": forward passes only (random probes), what an inference engine can do.
                y = x[0, 1:]
                miss = (1 - lb[:-1].softmax(-1).gather(-1, y[:, None]))
                g = layer_dirs(x, y)
                tgt_l = {}
                for l in layers:
                    d = g[l][:-1]
                    d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
                    t_ = torch.zeros(n, H, device=x.device)
                    t_[:-1] = a.gain * miss * d
                    tgt_l[l] = t_
            for l in layers:
                z = code(l, st["pre"][l][0])           # sparse code of the state before this layer's memory
                if a.target in ("grad", "spsa"):
                    err = tgt_l[l] / len(layers)
                elif a.target == "logit":
                    err = tgt / len(layers)
                else:
                    err = h_ctx[l] - st["cap"][l][0].float()   # what is still missing after the memory's output
                # positions 0..n-2 matter (they predict the next token); skip the attention sink token
                z, err = z[1:] * wpos[1:], err[1:]
                B[l] += a.eta * z.t() @ err
        with torch.no_grad():
            for l in layers:
                B[l].mul_(math.exp(-1 / a.tau_w))

    def own_logp(x, y):
        lg = model(x).logits[0, :-1].float().log_softmax(-1)
        return lg.gather(-1, y[:, None])[:, 0]

    def layer_dirs(x, y):
        n = x.shape[1]
        if a.target == "grad":
            with torch.enable_grad():
                st["delta"] = {l: torch.zeros(n, H, device=x.device, requires_grad=True) for l in layers}
                own_logp(x, y).sum().backward()
                g = {l: st["delta"][l].grad.detach()[0] if st["delta"][l].grad.dim() == 3 else st["delta"][l].grad.detach() for l in layers}
                st["delta"] = {}
            return g
        # SPSA with antithetic pairs; probe size relative to the residual norm at each layer
        st["delta"] = {}
        model(x)
        scale = {l: float(st["pre"][l][0].float().norm(dim=-1).mean()) / H ** 0.5 for l in layers}
        g = {l: torch.zeros(n, H, device=x.device) for l in layers}
        for _ in range(a.probes):
            v = {l: torch.randn(n, H, device=x.device) for l in layers}
            st["delta"] = {l: a.eps * scale[l] * v[l] for l in layers}
            fp = own_logp(x, y)
            st["delta"] = {l: -a.eps * scale[l] * v[l] for l in layers}
            fm = own_logp(x, y)
            diff = torch.zeros(n, device=x.device)
            diff[:-1] = (fp - fm) / 2
            for l in layers:
                g[l] += diff[:, None] * v[l]
        st["delta"] = {}
        return g

    def recall(o, d):
        pre = f"{DAYS[d][3:]}, Axel put the {OBJS[o]}"
        n_pre = ids(pre).shape[1]
        texts = [f"{pre} {pl}." for pl in PLACES]
        tok.padding_side = "right"
        enc = tok(texts, return_tensors="pt", padding=True)
        xx, m = enc.input_ids.cuda(), enc.attention_mask.cuda()
        with torch.no_grad():
            lp = model(xx, attention_mask=m).logits[:, :-1].float().log_softmax(-1)
        tl = lp.gather(-1, xx[:, 1:, None])[..., 0]
        mask = m[:, 1:].clone()
        mask[:, :n_pre - 1] = 0
        return int((tl * mask).sum(1).argmax())

    def nll(text):
        x = ids(text)
        with torch.no_grad():
            lg = model(x).logits[0, :-1].float()
        return float(nn.functional.cross_entropy(lg, x[0, 1:]))

    rng = random.Random(a.seed)
    combos = list(itertools.product(range(len(OBJS)), range(len(DAYS))))
    rng.shuffle(combos)
    events = [(o, rng.randrange(len(PLACES)), d) for o, d in combos[:a.n]]
    sent = lambda o, p, d: f"{DAYS[d][3:]}, Axel put the {OBJS[o]} {PLACES[p]}."
    # senses adapt to ordinary sentences of this kind first (the model has been alive before)
    for o, p, d in [(rng.randrange(len(OBJS)), rng.randrange(len(PLACES)), rng.randrange(len(DAYS))) for _ in range(100)]:
        adapt(sent(o, p, d))

    st["on"] = False
    base = [recall(o, d) == p for o, p, d in events]
    generic = ["The weather was nice, so we went for a walk in the park.",
               "Photosynthesis converts light energy into chemical energy.",
               "She opened the book and began to read the first chapter."]
    g0 = float(np.mean([nll(t) for t in generic]))
    st["on"] = True
    imm = []
    for i, e in enumerate(events):
        compile_fact(sent(*e))
        imm.append(recall(e[0], e[2]) == e[1])
    rec = np.array([recall(o, d) == p for o, p, d in events])
    g1 = float(np.mean([nll(t) for t in generic]))
    ages = np.array([len(events) - 1 - i for i in range(len(events))])
    rows = []
    for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200)]:
        s = (ages >= lo) & (ages < hi)
        if s.sum() >= 3:
            rows.append({"age": f"{lo}-{hi}", "recall": float(rec[s].mean()), "no_memory": float(np.array(base)[s].mean())})
    out = {"args": vars(a), "immediate_recall": float(np.mean(imm)), "rows": rows, "generic_nll": [g0, g1]}
    print(json.dumps(out, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"compile_{a.target}_L{a.layers.replace(',', '-')}_eta{a.eta}_it{a.iters}_s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="14")
    p.add_argument("--D", type=int, default=16384)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--tau_w", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kl_weight", type=int, default=1)
    p.add_argument("--target", default="ctx", choices=["ctx", "logit", "grad", "spsa"])
    p.add_argument("--probes", type=int, default=32)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--gain", type=float, default=30.0)
    p.add_argument("--no_prob", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_compile"))
    main(p.parse_args())
