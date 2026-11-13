"""Compiling consolidated facts into SLOW weights (U11 / Gemini round 7), local Qwen2.5-1.5B.

The fast memory injects, at the head input, the value v_t that makes the answer token y_{t+1} likely:
    v_t = dl * (1 - p_y) * (u_y - E_p[u]) / |u_y - E_p[u]|^2          (same rule as the LLM server fast weights)
Compiling = finding one dense matrix W (H x H) so that W h_t ~= v_t at the answer positions of the stored facts, with
h_t the model's own head-input state there; at run time the head input becomes h + W P h. P protects general
language: P = I - V0 V0^T with V0 the top-r0 principal directions of head-input states on general text (null-space
projection, the user's original theory). Closed form ridge: W = V (P H)^T ((P H)(P H)^T + lam I)^-1.
After compiling there is NO fast memory: the facts must come from W alone.

Measures: clean-anchor recall (C0), recall after a natural question (C1), general-text NLL (held-out), "Paris".
Conditions: n facts in {20, 100, 300}, r0 in {0, 512, 1024}.
"""
import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from anchor_mem2 import MODEL, NAMES, OBJS, PLACES, QFORMS

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES2 = NAMES + ["Ben", "Chloe", "Dan", "Emma", "Finn", "Grace", "Hugo", "Iris", "Jack", "Kate", "Liam", "Noah",
                  "Olivia", "Ava", "Lucas", "Mason", "Ella", "Zoe", "Ryan", "Nora"]
OBJS2 = OBJS + ["keys", "hat", "tablet", "lighter", "pen", "bike lock", "sunglasses", "flask", "backpack", "phone",
                "remote", "mug", "book", "jacket", "scissors", "tape"]
GENERAL_PROMPTS = ["The history of the Roman Empire", "How photosynthesis works", "A recipe for pancakes",
                   "The rules of football", "Why the sky is blue", "Tips for learning a new language",
                   "The life cycle of a butterfly", "How computers store data", "A short story about a lost dog",
                   "The geography of Japan", "How to change a tire", "The causes of the French Revolution",
                   "Basic principles of economics", "A letter to a friend about a holiday", "How vaccines work",
                   "The plot of a famous novel", "Instructions for planting tomatoes", "The physics of rainbows",
                   "A news report about a local festival", "How the stock market works"]


def main(a):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    H = model.config.hidden_size
    U = model.lm_head.weight.float()
    st = {"W": None, "cap": None}

    def hook(mod, inp, out):
        if st["cap"] is not None:
            st["cap"].append(out[0].float().detach())
        if st["W"] is not None:
            add = (out.float() @ st["W"].T).to(out.dtype)
            if a.gated:
                # only while recalling: rows at or after the anchor arrow (the thinker wrote "[x|y] ->"); during a
                # generate() step the single new row is always after the arrow
                g0 = st.get("gate_from")
                if g0 is None:
                    return out
                T = out.shape[1]
                mask = torch.zeros(1, T, 1, device=out.device, dtype=out.dtype)
                if T == 1:
                    mask[:] = 1
                else:
                    mask[:, g0:] = 1
                return out + add * mask
            return out + add
        return out

    model.model.norm.register_forward_hook(hook)

    # ---------------- general text: calibration (null space) and held-out NLL
    torch.manual_seed(0)
    gen = []
    for p in GENERAL_PROMPTS:
        ids = tok(p + ".", return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=120, do_sample=True, top_p=0.9, temperature=0.8)
        gen.append(tok.decode(out[0], skip_special_tokens=True))
    calib, held = gen[:14], gen[14:]

    def states(text):
        st["cap"] = []
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids)
        h = st["cap"][0]
        st["cap"] = None
        return ids[0], h

    Hc = torch.cat([states(t)[1] for t in calib])            # (N0, H)
    Hc = Hc - 0                                               # keep the mean: general text includes it
    _, _, Vt = torch.linalg.svd(Hc, full_matrices=False)       # rows of Vt = principal directions

    def nll(texts):
        tot, n = 0.0, 0
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                lg = model(ids).logits[0, :-1].float()
            tot += torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum").item(); n += ids.shape[1] - 1
        return tot / n

    def greedy(prompt, n_new=7):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        st["gate_from"] = ids.shape[1] - 1 if prompt.rstrip().endswith("->") else None
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n_new, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    base_nll = nll(held)
    results = []
    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES2 for o in OBJS2]
    rng.shuffle(pairs)
    for nf in a.n:
        facts = [(n_, o, rng.choice(PLACES)) for n_, o in pairs[:nf]]
        # targets: head-input states h_t and values v_t at the answer positions of every fact
        Hs, Vs = [], []
        for n_, o, p in facts:
            text = f"Memory notes.\n[{n_}|{o}] -> {p}."
            ids, h = states(text)
            pieces = [tok.decode(ids[:i + 1]) for i in range(len(ids))]
            a_end = max(i for i, s in enumerate(pieces) if s.endswith("->"))
            with torch.no_grad():
                lg = (h @ U.T)
            for t in range(a_end, len(ids) - 1):
                y = ids[t + 1].item()
                pr = torch.softmax(lg[t], -1)
                du = U[y] - pr @ U
                v = a.dl * (1 - pr[y].item()) * du / (du @ du + 1e-6)
                Hs.append(h[t]); Vs.append(v)
        Hm, Vm = torch.stack(Hs), torch.stack(Vs)              # (M, H)
        for r0 in a.r0:
            V0 = Vt[:r0].T if r0 > 0 else torch.zeros(H, 0, device="cuda")
            Pn = torch.eye(H, device="cuda") - V0 @ V0.T
            X = Hm @ Pn.T                                     # projected inputs (M, H)
            W = Vm.T @ X @ torch.linalg.inv(X.T @ X + a.lam * torch.eye(H, device="cuda"))   # (H, H)
            st["W"] = W @ Pn
            c0 = sum(greedy(f"Memory notes.\n[{n_}|{o}] ->").lower().startswith(p) for n_, o, p in facts) / nf
            c1 = sum(greedy(f"Memory notes.\n{rng.choice(QFORMS).format(n=n_, o=o)}\n[{n_}|{o}] ->").lower().startswith(p)
                     for n_, o, p in facts) / nf
            free = sum(p.split()[-1] in greedy(f"Q: Where is {n_}'s {o}?\nA:", 12).lower() for n_, o, p in facts) / nf
            g_nll = nll(held)
            paris = greedy("The capital of France is", 3)
            st["W"] = None
            row = {"gated": a.gated, "facts": nf, "r0": r0, "rank_free": H - r0, "C0": round(c0, 3), "C1": round(c1, 3),
                   "free_question": round(free, 3), "nll": round(g_nll, 4), "nll_base": round(base_nll, 4),
                   "paris": paris, "points": Hm.shape[0]}
            results.append(row)
            print(json.dumps(row), flush=True)
    os.makedirs(os.path.join(HERE, "results_compile"), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, "results_compile", f"compile_s{a.seed}{'_gated' if a.gated else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20, 100, 300])
    ap.add_argument("--r0", type=int, nargs="+", default=[0, 512, 1024])
    ap.add_argument("--dl", type=float, default=14.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gated", action="store_true")
    main(ap.parse_args())
