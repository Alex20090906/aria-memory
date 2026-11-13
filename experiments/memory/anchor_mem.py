"""Local model of the LLM server head-mode memory, to test two fixes before touching the engine.

Memory (same as LLM server head mode): key = k-WTA of the adapted state at a middle layer; value added to the final
hidden state right before the output matrix; write rule = exact u_y - E_p[u] direction, logit gain dl*(1-p_y).

Fixes under test
  integrated key (temporal context): scores s_t at the key layer are leaky-integrated over the sequence,
      s'_t = s_t + lam * s'_{t-1}, before top-k.  The key of "the" in "... -> in the red teapot" then still carries
      the anchor, so facts that share an answer prefix ("in the ...") stop colliding.  lam = 0 is the current engine.
  entity anchor: facts are stored as "[Leo|passport] -> in the red teapot."  and recalled from
      (C0) the same clean prompt "[Leo|passport] ->"               (exact prefix, the old test)
      (C1) a natural question before the anchor:  "Where did Leo leave the passport? [Leo|passport] ->"
           (the anchor sees a different context, so its key differs: a real generalisation test)
Recall = greedy continuation starts with the true place.
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
# many places share the answer prefix "in the" on purpose (the collision case)
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
    L = a.layer
    g = torch.Generator().manual_seed(a.seed)
    P = (torch.randn(H, a.D, generator=g) / H ** 0.5).cuda()
    B = torch.zeros(a.D, H, device="cuda")
    st = {"units": None, "mu": None, "sd": None, "bias": torch.zeros(a.D, device="cuda")}

    def key_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        x = nn.functional.layer_norm(h[0].float(), (H,))
        if st["mu"] is not None:
            x = (x - st["mu"]) / st["sd"]
        s = x @ P + st["bias"]
        if a.lam > 0:                                   # leaky integration over the sequence (temporal context)
            s = s.clone()
            for t in range(1, len(s)):
                s[t] = s[t] + a.lam * s[t - 1]
        st["units"] = s.topk(a.k, dim=-1).indices      # T x k
        st["raw"] = x
        return out

    def head_hook(mod, inp, out):                       # final norm output = the head's input
        if st.get("off"):
            return out
        z = torch.zeros(out.shape[1], a.D, device=out.device)
        z.scatter_(1, st["units"], 1 / a.k ** 0.5)
        return out + (z @ B).to(out.dtype)[None]

    model.model.layers[L].register_forward_hook(key_hook)
    model.model.norm.register_forward_hook(head_hook)
    U = model.lm_head.weight.float()
    ids = lambda t: tok(t, return_tensors="pt").input_ids.cuda()

    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES for o in OBJS]
    rng.shuffle(pairs)
    facts = [(n, o, rng.choice(PLACES)) for n, o in pairs[:a.n]]
    anchor = lambda n, o: f"[{n}|{o}] ->"

    # sensory adaptation + BCM-like bias on ordinary anchor sentences of other people/objects
    with torch.no_grad():
        st["off"] = True
        X = []
        for n, o in pairs[a.n:a.n + 80]:
            model(ids(f"{anchor(n, o)} {rng.choice(PLACES)}."))
            X.append(st["raw"])
        X = torch.cat(X)
        st["mu"], st["sd"] = X.mean(0), X.std(0) + 1e-3
        if a.bcm:
            for _ in range(5):
                cnt = torch.zeros(a.D, device="cuda")
                rows = 0
                for n, o in pairs[a.n:a.n + 80]:
                    model(ids(f"{anchor(n, o)} {rng.choice(PLACES)}."))
                    cnt += torch.bincount(st["units"].flatten(), minlength=a.D).float()
                    rows += len(st["units"])
                f = cnt / rows
                st["bias"] -= 0.3 * torch.log((f + 0.1 * a.k / a.D) / (a.k / a.D))
                st["bias"].clamp_(-3, 3)
        st["off"] = False

    @torch.no_grad()
    def write(text):
        x = ids(text)
        lg = model(x).logits[0].float()                 # memory on: the delta rule writes only what is missing
        units = st["units"]
        n_pre = ids(text.split("->")[0] + "->").shape[1]
        p = lg[:-1].softmax(-1)
        for t in range(n_pre - 1, x.shape[1] - 1):      # positions from the anchor end on predict the answer
            y = x[0, t + 1]
            du = U[y] - p[t] @ U
            tgt = a.dl * (1 - p[t, y]) * du / (du @ du + 1e-6)
            B[units[t]] += tgt / a.k ** 0.5

    @torch.no_grad()
    def recall(prompt, n_new=8):
        x = ids(prompt)
        for _ in range(n_new):
            nxt = model(x).logits[0, -1].argmax()
            x = torch.cat([x, nxt.view(1, 1)], 1)
        return tok.decode(x[0, ids(prompt).shape[1]:])

    for n, o, p in facts:
        write(f"{anchor(n, o)} {p}.")
    res = {"C0": [], "C1": []}
    for n, o, p in facts:
        ans0 = recall(anchor(n, o))
        q = rng.choice(QFORMS).format(n=n, o=o)
        ans1 = recall(f"{q} {anchor(n, o)}")
        res["C0"].append(ans0.strip().startswith(p))
        res["C1"].append(ans1.strip().startswith(p))
    st["off"] = True
    base = np.mean([recall(anchor(n, o)).strip().startswith(p) for n, o, p in facts])
    out = {"args": vars(a), "C0": float(np.mean(res["C0"])), "C1": float(np.mean(res["C1"])), "no_memory": float(base)}
    print(json.dumps(out), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"anchor_L{a.layer}_lam{a.lam}_bcm{a.bcm}_n{a.n}_s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--bcm", type=int, default=1)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--D", type=int, default=16384)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--dl", type=float, default=14.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_anchor"))
    main(p.parse_args())
