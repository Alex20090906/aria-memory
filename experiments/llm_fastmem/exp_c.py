"""Step C on LLM server: anchor memory with natural questions.
Writes N facts "[owner|object] -> place." then recalls each from
  C0  PRE + "[owner|object] ->"                       (clean anchor)
  C1  PRE + "<natural question>\n[owner|object] ->"   (the question changes every hidden state before the anchor)
and checks that a general question is unaffected.  Writes exp_c.json."""
import argparse
import json
import os
import random

import numpy as np

import fmem_anchor as fa
import fmem_tool as f

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Axel", "Ken", "Lena", "Ivan", "Rosa", "Pablo", "Hana"]
OBJS = ["passport", "spare key", "watch", "glasses", "ticket", "charger", "ring", "wallet", "umbrella", "headphones",
        "notebook", "camera", "scarf", "medicine", "bank card", "laptop"]
PLACES = ["in the red teapot", "in the blue envelope", "in the garage drawer", "in the freezer", "in the toolbox",
          "in the rice jar", "in the old boot", "in the piano bench", "in the laundry basket", "in the glovebox",
          "under the doormat", "under the piano", "behind the bookshelf", "behind the mirror", "on the windowsill",
          "on top of the fridge"]
QFORMS = ["Where did {n} leave the {o}?", "Do you know where {n}'s {o} is?", "I can't find {n}'s {o}, any idea?",
          "Where is the {o} that {n} had?", "We are late, where on earth is {n}'s {o}?"]


def main(a):
    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES for o in OBJS]
    rng.shuffle(pairs)
    facts = [(n, o, rng.choice(PLACES)) for n, o in pairs[:a.n]]
    f.reset(None)
    fa.off()
    generic_q = "What is the capital of France? Answer in one word."
    g0 = f.chat([{"role": "user", "content": generic_q}], max_tokens=8)
    for n, o, p in facts:
        k_ = fa.write(n, o, p, dl=a.dl)
        print("wrote", n, o, p, k_, flush=True)
    res = {"C0": [], "C1": [], "rows": []}
    for n, o, p in facts:
        r0 = fa.recall(f.PRE + f"[{n}|{o}] ->", n, o).strip()
        q = rng.choice(QFORMS).format(n=n, o=o)
        r1 = fa.recall(f.PRE + q + "\n" + f"[{n}|{o}] ->", n, o).strip()
        ok0, ok1 = r0.lower().startswith(p), r1.lower().startswith(p)
        res["C0"].append(ok0); res["C1"].append(ok1)
        res["rows"].append({"fact": f"{n}'s {o}: {p}", "C0": r0[:40], "C1": r1[:40], "q": q})
        print(ok0, ok1, "|", p, "|", r0[:30], "|", r1[:30], flush=True)
    fa.off()
    g1 = f.chat([{"role": "user", "content": generic_q}], max_tokens=8)
    out = {"n": a.n, "dl": a.dl, "C0": float(np.mean(res["C0"])), "C1": float(np.mean(res["C1"])),
           "generic": [g0, g1], "rows": res["rows"]}
    print("C0", out["C0"], "C1", out["C1"], "generic", g0, "->", g1)
    json.dump(out, open(os.path.join(HERE, f"exp_c_n{a.n}_dl{a.dl}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--dl", type=float, default=14.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
