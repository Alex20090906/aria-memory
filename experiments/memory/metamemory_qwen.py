"""H14b / U16 question 5: which content-confidence signal separates a true memory from a corrupted one?

Local model of the LLM server memory (Qwen2.5-1.5B, same design as anchor_mem2 with key=bilinear, ans=shift and no
previous token in the answer keys): values added at the head input with the exact write rule, theta familiarity on the
anchor units. Facts use REALISTIC placements (U17: every object has 5 places people really use) and the no-memory
answer is recorded, so common-sense guesses are not counted as memory.

Load is pushed past saturation (n facts >> D/k) so that confident-wrong answers appear. For every recall that passes the
familiarity gate we record candidate confidence signals and then ask which one best separates correct from wrong:
  p1      probability of the first answer token with memory (the current H14 check)
  pmin    minimum over the answer tokens of that probability (weakest link)
  sir     coherence of the k unit rows at each answer step: |sum w|^2 / (k sum |w|^2), min over steps
  gain    log-odds the memory adds to the chosen token: logit_mem - logit_nomem, min over steps
  theta   mean familiarity of the anchor units
Output: AUC of each signal (correct vs wrong among gate-passing answers) and, for each signal, the threshold that keeps
90% of the correct answers and the share of wrong answers it removes.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NAMES = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Axel", "Ken", "Lena", "Ivan", "Rosa", "Pablo", "Hana",
         "Ben", "Chloe", "Dan", "Emma", "Finn", "Grace", "Hugo", "Iris", "Jack", "Kate", "Liam", "Noah", "Olivia", "Ava",
         "Lucas", "Mason", "Ella", "Zoe", "Ryan", "Nora", "Oscar", "Ruby", "Sam", "Tara", "Uma", "Vera"]
PLACES = {
    "passport": ["in the desk drawer", "in the bedroom safe", "in the document folder", "in the suitcase pocket", "in the filing cabinet"],
    "spare key": ["under the doormat", "in the kitchen drawer", "on the key hook", "in the flower pot", "at the neighbor's house"],
    "watch": ["on the nightstand", "in the jewelry box", "on the bathroom shelf", "in the desk drawer", "on the dresser"],
    "glasses": ["on the nightstand", "in the glasses case", "on the desk", "on the bathroom counter", "next to the sofa"],
    "ticket": ["in the wallet", "in the desk drawer", "on the fridge door", "in the backpack", "in the coat pocket"],
    "charger": ["in the desk drawer", "next to the bed", "in the backpack", "on the kitchen counter", "in the living room outlet"],
    "ring": ["in the jewelry box", "on the bathroom shelf", "on the nightstand", "in the dresser drawer", "in the soap dish"],
    "wallet": ["on the entryway table", "in the coat pocket", "in the backpack", "on the desk", "in the car"],
    "umbrella": ["in the umbrella stand", "next to the front door", "in the car", "in the backpack", "on the balcony"],
    "headphones": ["in the backpack", "on the desk", "next to the bed", "in the desk drawer", "on the sofa"],
    "notebook": ["on the desk", "in the backpack", "on the bookshelf", "in the desk drawer", "on the nightstand"],
    "camera": ["in the camera bag", "on the bookshelf", "in the closet", "in the desk drawer", "in the backpack"],
    "scarf": ["in the closet", "on the coat rack", "in the dresser drawer", "in the coat pocket", "on the chair"],
    "medicine": ["in the medicine cabinet", "in the kitchen drawer", "on the nightstand", "in the fridge", "in the first aid box"],
    "bank card": ["in the wallet", "in the desk drawer", "in the phone case", "in the document folder", "in the coat pocket"],
    "laptop": ["on the desk", "in the backpack", "on the sofa", "on the dining table", "in the laptop bag"],
    "keys": ["on the key hook", "on the entryway table", "in the coat pocket", "in the backpack", "on the kitchen counter"],
    "phone": ["on the nightstand", "on the desk", "on the sofa", "in the coat pocket", "on the kitchen counter"],
    "remote": ["on the sofa", "on the coffee table", "next to the TV", "on the armchair", "in the TV cabinet drawer"],
    "book": ["on the bookshelf", "on the nightstand", "on the desk", "in the backpack", "on the sofa"],
    "jacket": ["in the closet", "on the coat rack", "on the chair", "in the car", "behind the door"],
    "scissors": ["in the kitchen drawer", "in the desk drawer", "in the pencil cup", "in the sewing box", "in the toolbox"],
    "flashlight": ["in the kitchen drawer", "in the toolbox", "next to the bed", "in the car", "in the storage cabinet"],
    "helmet": ["in the garage", "on the bike", "next to the front door", "in the closet", "on the scooter"],
}
QFORMS = ["Where did {n} leave the {o}?", "Do you know where {n}'s {o} is?", "I can't find {n}'s {o}, any idea?",
          "Where is the {o} that {n} had?"]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(gt)


def main(a):
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    H = model.config.hidden_size
    D, k = a.D, a.k
    g = torch.Generator().manual_seed(a.seed)
    P = (torch.randn(H, D, generator=g) / H ** 0.5).cuda()
    B = torch.zeros(D, H, device="cuda")
    theta = torch.zeros(D, device="cuda")
    # U16 question 2 (key echo): each anchor also stores its own random +-1 signature in an echo bank on the same
    # units; familiarity = cosine between the summed echo rows and the query's signature. Unlike theta (a scalar per
    # unit that saturates once every unit has been touched), the echo of a never-heard label is random noise.
    EB = torch.zeros(D, a.Le, device="cuda")

    def sigvec(label):
        gs = torch.Generator(device="cuda").manual_seed(abs(hash(label)) % (2 ** 31))
        return (torch.randint(0, 2, (a.Le,), generator=gs, device="cuda").float() * 2 - 1)

    def echo(anc, label):
        z = EB[anc].sum(0)
        return float(z @ sigvec(label) / (z.norm() * a.Le ** 0.5 + 1e-6))
    V = model.lm_head.weight.shape[0]
    if a.ecc:
        # U16 question 1 (error-correcting code, digital storage in the same units): every token id has a random
        # +-1 codeword of length L; the units store the sum of codewords; recall = nearest codeword (cleanup) with a
        # margin that doubles as the confidence; a clean, strong push for the decoded token replaces the analog value
        gc = torch.Generator(device="cuda").manual_seed(1234)
        CODES = (torch.randint(0, 2, (V, a.L), generator=gc, device="cuda", dtype=torch.int8) * 2 - 1).half()
        C = torch.zeros(D, a.L, device="cuda")
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
    norm = lambda v: nn.functional.layer_norm(v.float(), (H,))

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
        obj = list(range(bar + 1, rb)) or [bar]
        return a_end, subj, obj

    def anchor_units(x):
        hs = hstates(x)
        a_end, subj, obj = spans(x)
        xs = (norm(hs[a.Ls][0, subj]).mean(0) - stats["s"][0]) / stats["s"][1]
        xo = (norm(hs[a.Ls][0, obj]).mean(0) - stats["o"][0]) / stats["o"][1]
        s = (xs @ P) * torch.roll(xo @ P, 1)
        return s.topk(k).indices, a_end

    def units_for(x):
        anc, a_end = anchor_units(x)
        out = {a_end: anc}
        for t in range(a_end + 1, x.shape[1]):
            out[t] = (anc + (t - a_end) * 1000003) % D        # answer keys without the previous token
        return out, anc, a_end

    rng = random.Random(a.seed)
    objs = list(PLACES)
    pairs = [(n, o) for n in NAMES for o in objs]
    rng.shuffle(pairs)
    anchor = lambda n, o: f"[{n}|{o}] ->"

    acc = {"s": [], "o": []}
    for i, (n, o) in enumerate(pairs[-120:]):
        txt = f"{anchor(n, o)} {rng.choice(PLACES[o])}."
        if i % 2:
            txt = rng.choice(QFORMS).format(n=n, o=o) + "\n" + txt
        x = ids(txt)
        hs = hstates(x)
        _, subj, obj = spans(x)
        acc["s"].append(norm(hs[a.Ls][0, subj]).mean(0)[None])
        acc["o"].append(norm(hs[a.Ls][0, obj]).mean(0)[None])
    for key_, v in acc.items():
        V = torch.cat(v)
        stats[key_] = (V.mean(0), V.std(0) + 1e-3)

    @torch.no_grad()
    def write(n, o, p):
        x = ids(f"{anchor(n, o)} {p}.")
        u, anc, _ = units_for(x)
        theta.mul_(math.exp(-1 / a.theta_tau))
        theta[anc] = 1.0
        EB[anc] += sigvec(f"{n}|{o}")
        st["units"] = u
        lg = model(x).logits[0].float()      # error-driven: targets read with the current memory injected
        pr = lg[:-1].softmax(-1)
        for t, units in u.items():
            if t + 1 >= x.shape[1]:
                continue
            y = x[0, t + 1]
            du = U[y] - pr[t] @ U
            B[units] += (a.dl * (1 - pr[t, y]) * du / (du @ du + 1e-6)) / k ** 0.5
            if a.ecc:
                C[units] += CODES[y].float() / k ** 0.5

    @torch.no_grad()
    def recall(prompt, n_new=6):
        x = ids(prompt)
        n0 = x.shape[1]
        sig = {"p": [], "sir": [], "gain": []}
        for _ in range(n_new):
            u, anc, a_end = units_for(x)
            if a.ecc:
                z = C[u[x.shape[1] - 1]].sum(0)
                sc = (CODES.float() @ z)
                top2 = sc.topk(2)
                y = int(top2.indices[0])
                margin = float((top2.values[0] - top2.values[1]) / (z.norm() * a.L ** 0.5 + 1e-6))
                sig["p"].append(margin); sig["gain"].append(float(top2.values[0] / (z.norm() * a.L ** 0.5 + 1e-6)))
                sig["sir"].append(margin)
                x = torch.cat([x, torch.tensor([[y]], device="cuda")], 1)
                if "." in tok.decode([y]) or chr(10) in tok.decode([y]):
                    break
                continue
            st["units"] = u
            lg = model(x).logits[0, -1].float()
            st["off"] = True
            lg0 = model(x).logits[0, -1].float()
            st["off"] = False
            y = int(lg.argmax())
            sig["p"].append(float(lg.softmax(-1)[y]))
            sig["gain"].append(float(lg[y] - lg0[y]))
            w = B[u[x.shape[1] - 1]]
            sig["sir"].append(float((w.sum(0) ** 2).sum() / (k * (w ** 2).sum() + 1e-12)))
            x = torch.cat([x, torch.tensor([[y]], device="cuda")], 1)
            if "." in tok.decode([y]) or "\n" in tok.decode([y]):
                break
        _, anc, _ = units_for(ids(prompt))
        return tok.decode(x[0, n0:]), sig, float(theta[anc].mean())

    @torch.no_grad()
    def prior(prompt, n_new=6):
        # common-sense guess without memory: two worked examples show the format, so the model answers with a place
        st["off"] = True
        prompt = ("Memory notes.\n[Ann|cup] -> in the kitchen cabinet.\n[Bob|hat] -> on the coat rack.\n" + prompt)
        x = ids(prompt)
        out = model.generate(x, max_new_tokens=n_new, do_sample=False)
        st["off"] = False
        return tok.decode(out[0, x.shape[1]:])

    results = []
    facts = [(n, o, rng.choice(PLACES[o])) for n, o in pairs[:a.n]]
    never = pairs[a.n:a.n + a.never]
    for i, (n, o, p) in enumerate(facts):
        write(n, o, p)
    for kind, items in (("told", facts), ("never", [(n, o, None) for n, o in never])):
        for idx, (n, o, p) in enumerate(items):
            prompt = f"{rng.choice(QFORMS).format(n=n, o=o)}\n{anchor(n, o)}"
            _, anc, _ = units_for(ids(prompt))
            th = float(theta[anc].mean())
            row = {"kind": kind, "fact": f"{n}|{o}", "truth": p, "theta": th, "echo": echo(anc, f"{n}|{o}"),
                   "age": (a.n - idx) if kind == "told" else None}
            if th >= a.gate:
                ans, sig, _ = recall(prompt)
                row.update({"answer": ans.strip(), "p1": sig["p"][0], "pmin": min(sig["p"]), "sir": min(sig["sir"]),
                            "gain": min(sig["gain"]), "correct": bool(p) and ans.strip().lower().startswith(p)})
            else:
                row.update({"answer": "", "correct": False})
            if p is not None:
                row["prior"] = prior(prompt).strip()
                row["prior_correct"] = row["prior"].lower().startswith(p)
            results.append(row)

    told = [r for r in results if r["kind"] == "told"]
    answered = [r for r in told if r["answer"]]
    ok = [r for r in answered if r["correct"]]
    bad = [r for r in answered if not r["correct"]]
    nev = [r for r in results if r["kind"] == "never"]
    summ = {"n": a.n, "D": D, "k": k, "told": len(told), "answered": len(answered), "correct": len(ok),
            "confident_wrong": len(bad), "gate_closed": len(told) - len(answered),
            "prior_correct": sum(r["prior_correct"] for r in told),
            "never_passed_gate": sum(1 for r in nev if r["answer"]), "never": len(nev), "signals": {}}
    # familiarity quality: told vs never-heard labels (gate quality), for theta and for the echo signature
    summ["familiarity_auc_told_vs_never"] = {fam: auc([r[fam] for r in told], [r[fam] for r in nev]) for fam in ("theta", "echo")}
    ecs = sorted(r["echo"] for r in nev)
    if ecs:
        th_e = ecs[int(0.95 * (len(ecs) - 1))]      # echo gate that lets at most 5% of never-heard labels through
        summ["echo_gate95"] = {"threshold": th_e, "told_pass": float(np.mean([r["echo"] > th_e for r in told])),
                               "told_pass_correct": float(np.mean([r["echo"] > th_e for r in told if r.get("correct")] or [0])),
                               "told_pass_wrong": float(np.mean([r["echo"] > th_e for r in told if r["answer"] and not r["correct"]] or [0]))}
    for s in ("p1", "pmin", "sir", "gain", "theta", "echo"):
        pos = [r[s] for r in ok]
        neg = [r[s] for r in bad]
        d = {"auc": auc(pos, neg)}
        if pos and neg:
            th = float(np.quantile(pos, 0.10))           # keep 90% of the correct answers
            d["keep90_threshold"] = th
            d["wrong_removed"] = float(np.mean([v < th for v in neg]))
        summ["signals"][s] = d
    print(json.dumps(summ, indent=1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"summary": summ, "rows": results}, open(os.path.join(a.out, f"meta_n{a.n}_D{D}_s{a.seed}{'_ecc' if a.ecc else ''}.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--never", type=int, default=60)
    p.add_argument("--D", type=int, default=4096)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--Ls", type=int, default=2)
    p.add_argument("--dl", type=float, default=14.0)
    p.add_argument("--gate", type=float, default=0.5)
    p.add_argument("--theta_tau", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ecc", action="store_true")
    p.add_argument("--L", type=int, default=256)
    p.add_argument("--Le", type=int, default=128)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_meta"))
    main(p.parse_args())
