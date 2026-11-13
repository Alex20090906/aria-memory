"""H2: two levels of familiarity.

item  level: "did exactly this happen?"  (deep layer, strict sparse code)
topic level: "was this thing talked about?" (shallower layer and/or looser code)

Store N gist facts ("Leo's passport is under the doormat."). Probes:
  topic question   "Where is Leo's passport?"      for a stored fact   vs  for a never-mentioned owner+object
  similar lure     "Leo's passport is in the car." (same topic, wrong detail)  vs  the true fact
Prediction: a topic code separates stored-topic questions from new-topic questions (AUC > 0.8) while an item
code does not (< 0.6); the item code separates true facts from similar lures, the topic code does not.
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
NAMES = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Axel", "Ken", "Lena", "Ivan", "Rosa"]
OBJS = ["passport", "spare key", "watch", "glasses", "ticket", "charger", "ring", "wallet", "umbrella", "headphones",
        "notebook", "camera", "scarf", "medicine", "remote control", "bank card"]
PLACES = ["under the doormat", "in the red teapot", "in the bathroom cabinet", "behind the bookshelf", "in the blue envelope",
          "in the garage drawer", "under the piano", "in the freezer", "in the car", "on the windowsill", "in the closet"]


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main(a):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    H = model.config.hidden_size
    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES for o in OBJS]
    rng.shuffle(pairs)
    stored, new = pairs[:a.n], pairs[a.n:2 * a.n]
    facts = [(n, o, rng.choice(PLACES)) for n, o in stored]

    @torch.no_grad()
    def states(text):
        out = model(tok(text, return_tensors="pt").input_ids.cuda(), output_hidden_states=True)
        return {l: nn.functional.layer_norm(out.hidden_states[l][0, 1:].float(), (H,)).mean(0) for l in a.layers}

    # senses adapt to sentences of this kind (unrelated owners/objects)
    warm = [states(f"{rng.choice(NAMES)}'s {rng.choice(OBJS)} is {rng.choice(PLACES)}.") for _ in range(100)]
    warm += [states(f"Where is {rng.choice(NAMES)}'s {rng.choice(OBJS)}?") for _ in range(50)]
    mu = {l: torch.stack([w[l] for w in warm]).mean(0) for l in a.layers}
    sd = {l: torch.stack([w[l] for w in warm]).std(0) + 1e-3 for l in a.layers}
    g = torch.Generator().manual_seed(7)
    P = {l: (torch.randn(H, a.D, generator=g) / H ** 0.5).cuda() for l in a.layers}

    def code(l, h, k):
        s = ((h - mu[l]) / sd[l]) @ P[l]
        c = torch.zeros_like(s)
        c[s.topk(k).indices] = 1 / k ** 0.5
        return c

    St = {i: states(f"{n}'s {o} is {p}.") for i, (n, o, p) in enumerate(facts)}
    Q_old = [states(f"Where is {n}'s {o}?") for n, o in stored]
    Q_new = [states(f"Where is {n}'s {o}?") for n, o in new]
    T = [states(f"{n}'s {o} is {p}.") for n, o, p in facts]
    L = [states(f"{n}'s {o} is {rng.choice([q for q in PLACES if q != p])}.") for n, o, p in facts]
    rows = []
    for l in a.layers:
        for k in a.ks:
            C = torch.stack([code(l, St[i][l], k) for i in St])
            fam = lambda s: float(((C @ code(l, s[l], k)) ** 2).sum())
            rows.append({"layer": l, "k": k,
                         "topic_auc_question_old_vs_new": auc([fam(s) for s in Q_old], [fam(s) for s in Q_new]),
                         "item_auc_true_vs_lure": auc([fam(s) for s in T], [fam(s) for s in L])})
            print(rows[-1], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "rows": rows}, open(os.path.join(a.out, f"two_level_s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 14, 20])
    p.add_argument("--ks", type=int, nargs="+", default=[20, 100, 400])
    p.add_argument("--D", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results_two_level"))
    main(p.parse_args())
