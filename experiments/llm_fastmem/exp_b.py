"""Step B (H5 capacity) on the AI PC: write N facts in order, then cue-recall every one; with the current BCM
bias and with the bias set to zero.  Recall is reported by age (how many facts were written after it).
Writes exp_b.json."""
import argparse
import json
import os
import random

import numpy as np

import fmem_tool as f

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Axel", "Ken", "Lena", "Ivan", "Rosa", "Pablo", "Hana"]
OBJS = ["passport", "spare key", "watch", "glasses", "ticket", "charger", "ring", "wallet", "umbrella", "headphones",
        "notebook", "camera", "scarf", "medicine", "bank card", "laptop"]
PLACES = ["under the doormat", "inside the red teapot", "in the bathroom cabinet", "behind the bookshelf",
          "in the blue envelope", "in the garage drawer", "under the piano", "in the freezer", "on top of the fridge",
          "inside the old boot", "under the sofa cushion", "behind the mirror", "in the toolbox", "on the windowsill",
          "inside the rice jar", "under the bathroom sink"]


def bias(set_zero, saved=None):
    n_mem, D, k, n, layers = f.read_header()
    lay, _ = f.mem_layout(n_mem, D, n)
    mm = np.memmap(f.MEM, dtype=np.uint8, mode="r+")
    b = np.ndarray((D,), np.float32, mm, lay[0][4])
    old = b.copy()
    if set_zero:
        b[:] = 0
    elif saved is not None:
        b[:] = saved
    mm.flush(); del mm, b
    f.touch_mem()
    return old


def main(a):
    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES for o in OBJS]
    rng.shuffle(pairs)
    facts = [(n, o, rng.choice(PLACES)) for n, o in pairs[:a.n]]
    saved = bias(False, None)          # read the current (BCM) bias without changing it
    res = {}
    for cond in a.conds.split(","):
        if cond == "nobcm":
            bias(True)
        else:
            bias(False, saved)
        f.reset(None)
        for nm, o, p in facts:
            f.write_head(argparse.Namespace(text=f"{nm}'s {o} is {p}.", dl=a.dl, tau_w=a.tau_w))
        rows = []
        for i, (nm, o, p) in enumerate(facts):
            ans = f.chat([{"role": "user", "content": f.RAW + f.PRE + f"{nm}'s {o} is"}], max_tokens=8)
            ok = ans.strip().lower().startswith(p.lower()) or p.split()[-1].lower() in ans.lower()
            rows.append({"age": a.n - 1 - i, "ok": bool(ok), "ans": ans.strip()[:40], "truth": p})
        ages = np.array([r["age"] for r in rows]); ok = np.array([r["ok"] for r in rows])
        bins = {}
        for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80)]:
            s = (ages >= lo) & (ages < hi)
            if s.sum():
                bins[f"{lo}-{hi}"] = round(float(ok[s].mean()), 3)
        res[cond] = {"overall": round(float(ok.mean()), 3), "by_age": bins, "rows": rows,
                     "generic": f.chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], max_tokens=10)}
        print(cond, res[cond]["overall"], bins, res[cond]["generic"], flush=True)
    bias(False, saved)
    json.dump(res, open(os.path.join(HERE, f"exp_b_n{a.n}_dl{a.dl}_tau{a.tau_w}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--dl", type=float, default=14.0)
    p.add_argument("--tau_w", type=float, default=1e9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conds", default="bcm,nobcm")
    main(p.parse_args())
