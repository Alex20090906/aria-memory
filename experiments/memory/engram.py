"""Memory model prototype: three-component synapse with a familiarity engram.

Mapping of the user's theory:
  w     explicit weight, stores details (cue -> place), decays fast (tau_w)
  theta engram on co-activation synapses, stores only "this combination happened",
        decays slowly (tau_theta), no details
  tau   per-store time constants (fixed here)

Open problem tested: can familiarity be separated from semantic similarity?
  dense engram : theta lives on the 384-d sentence embedding  -> familiarity = similarity?
  sparse engram: theta lives on a 8192-d k-winners-take-all expansion (pattern separation,
                 like the dentate gyrus) -> similar-but-new events get different codes?

Protocol: a stream of N events "Axel put the {obj} {place} on {day}". After the stream,
probe each stored event at its age (steps since it was stored):
  recall     : from the cue "Axel put the {obj} ... on {day}", read the place from w
  familiarity: score the full sentence; compare old events vs
               similar lures (same obj and day, different place: never happened) and
               unrelated lures (object never seen).
"""
import argparse
import itertools
import json
import os
import random

import numpy as np
import torch

OBJS = ["keys", "wallet", "phone", "passport", "glasses", "umbrella", "laptop", "headphones",
        "notebook", "charger", "camera", "watch", "scarf", "ring", "ticket", "medicine",
        "remote control", "flashlight", "backpack", "water bottle", "spare battery", "hat",
        "gloves", "earrings", "letter", "photo album", "toolbox", "teddy bear", "guitar pick", "map"]
PLACES = ["in the kitchen drawer", "under the bed", "on the bookshelf", "in the car",
          "in the blue box", "behind the sofa", "on the windowsill", "in the closet",
          "in the garage", "next to the TV"]
DAYS = ["on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday", "on Saturday",
        "on Sunday"]


def sentence(o, p, d):
    return f"Axel put the {o} {p} {d}."


def cue(o, d):
    return f"Where did Axel put the {o} {d}?"


class Engram:
    def __init__(self, d, n_places, tau_w, tau_theta, sparse, D, k, device, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = torch.zeros(d, n_places, device=device)
        self.tau_w, self.tau_theta = tau_w, tau_theta
        self.sparse, self.k = sparse, k
        if sparse:
            self.P = torch.randn(d, D, generator=g).to(device) / d ** 0.5
            self.W = torch.zeros(D, n_places, device=device)
        # engram stored as its codes + strengths: c^T Theta c == sum_i s_i (c . c_i)^2
        self.codes, self.strength = [], []

    def code(self, x):
        if not self.sparse:
            return x
        h = x @ self.P
        c = torch.zeros_like(h)
        c[h.topk(self.k).indices] = 1.0
        return c / self.k ** 0.5

    def step(self):
        self.W.mul_(np.exp(-1 / self.tau_w))
        self.strength = [x * np.exp(-1 / self.tau_theta) for x in self.strength]

    def store(self, cue_vec, place_idx, event_vec):
        # explicit weight: delta rule on (pattern-separated) cue -> one-hot place
        cue_vec = self.code(cue_vec)
        target = torch.zeros(self.W.shape[1], device=self.W.device)
        target[place_idx] = 1.0
        err = target - cue_vec @ self.W
        self.W += torch.outer(cue_vec, err)
        # engram: co-activation trace, no details kept
        self.codes.append(self.code(event_vec))
        self.strength.append(1.0)

    def recall(self, cue_vec):
        return int((self.code(cue_vec) @ self.W).argmax())

    def familiarity(self, event_vec):
        c = self.code(event_vec)
        C = torch.stack(self.codes)
        return float(((C @ c) ** 2 * torch.tensor(self.strength, device=c.device)).sum())


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def run(args):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    rng = random.Random(args.seed)

    combos = list(itertools.product(range(len(OBJS)), range(len(DAYS))))
    rng.shuffle(combos)
    combos = combos[: args.n]
    seen_objs = {o for o, _ in combos}
    events = [(o, rng.randrange(len(PLACES)), d) for o, d in combos]
    # half the objects never appear -> unrelated lures
    unseen = [o for o in range(len(OBJS)) if o not in seen_objs]

    texts = [sentence(OBJS[o], PLACES[p], DAYS[d]) for o, p, d in events]
    cues = [cue(OBJS[o], DAYS[d]) for o, p, d in events]
    sim_lures = [sentence(OBJS[o], PLACES[(p + 1 + rng.randrange(len(PLACES) - 1)) % len(PLACES)], DAYS[d])
                 for o, p, d in events]
    if unseen:
        unrel = [sentence(OBJS[rng.choice(unseen)], PLACES[rng.randrange(len(PLACES))], DAYS[rng.randrange(len(DAYS))])
                 for _ in events]
    else:
        unrel = [sentence("violin", PLACES[rng.randrange(len(PLACES))], DAYS[rng.randrange(len(DAYS))]) for _ in events]

    E = lambda xs: torch.tensor(enc.encode(xs, normalize_embeddings=True), device=args.device)
    ev, cv, sl, ul = E(texts), E(cues), E(sim_lures), E(unrel)
    cos_sim_lure = float((ev * sl).sum(1).mean())

    results = {}
    for name, sparse in (("dense", False), ("sparse", True)):
        m = Engram(ev.shape[1], len(PLACES), args.tau_w, args.tau_theta, sparse, args.D, args.k, args.device, args.seed)
        for i in range(len(events)):
            m.step()
            m.store(cv[i], events[i][1], ev[i])
        ages = np.array([len(events) - 1 - i for i in range(len(events))])
        rec = np.array([m.recall(cv[i]) == events[i][1] for i in range(len(events))])
        f_old = np.array([m.familiarity(ev[i]) for i in range(len(events))])
        f_sim = np.array([m.familiarity(sl[i]) for i in range(len(events))])
        f_unr = np.array([m.familiarity(ul[i]) for i in range(len(events))])
        bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10 ** 9)]
        rows = []
        for lo, hi in bins:
            sel = (ages >= lo) & (ages < hi)
            if sel.sum() < 3:
                continue
            rows.append({
                "age": f"{lo}-{hi if hi < 10**9 else 'max'}", "n": int(sel.sum()),
                "recall": float(rec[sel].mean()),
                "auc_vs_similar": auc(f_old[sel], f_sim[sel]),
                "auc_vs_unrelated": auc(f_old[sel], f_unr[sel]),
            })
        results[name] = rows
    out = {"args": vars(args), "cos_event_vs_similar_lure": cos_sim_lure, "chance_recall": 1 / len(PLACES),
           "results": results}
    print(json.dumps(out, indent=1))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"engram_n{args.n}_k{args.k}_s{args.seed}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--tau_w", type=float, default=40.0)
    p.add_argument("--tau_theta", type=float, default=2000.0)
    p.add_argument("--D", type=int, default=8192)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    run(p.parse_args())
