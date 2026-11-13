"""Three-model integration, first end-to-end run.

  model 1  always-on GRU (world3, Qwen3-Embedding, own familiarity engram), one action per tick
  model 2  memory model v1 inside a local Qwen2.5-1.5B: gist -> sparse fast weights w + theta traces
  model 3  LLM server thinker (remote), reaches memory only through the `recall` tool

Memory write policy (the experimental variable):
  gated : an event is written only if model 1 did something with it (action != WAIT)
          -> attention decides what is worth remembering
  all   : every event is written (ambient noise, TV, ...)
  none  : thinker without memory (baseline)

Questions: people tell Aria where they put things, later ask. Each answer is scored as
correct / honest "don't remember" / wrong (an invented detail).
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model1"))
sys.path.insert(0, os.path.join(ROOT, "integration"))
import world3  # noqa: E402
import train_gru2 as tg  # noqa: E402
from memory_v1 import MemoryModel  # noqa: E402
from thinker import think  # noqa: E402

UNKNOWN = re.compile(r"(don[’']t|do not|can[’']t|cannot|couldn[’']t|could not)\s+(remember|recall|find|know)|not sure|"
                     r"no memory|not certain|doesn[’']t come|unsure|don[’']t have|no (real |specific )?memory", re.I)


def to_memory_text(ev):
    tag, body = ev.split("] ", 1)
    if tag == "[CAM":
        return f"I saw: {body}"
    if tag == "[MIC":
        who, said = body.split(": ", 1) if ": " in body else ("Someone", body)
        return f"{who} said: {said.replace('Aria, ', '').replace(', Aria', '')}"
    return None


def model1_actions(ticks, ckpt):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(tg.ENCODERS["qwen3emb"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    tg.EMB = enc.get_sentence_embedding_dimension()
    X, _ = tg.featurize(ticks, {}, enc, seed=1)
    m = tg.Model1(X.shape[1]).cuda()
    m.load_state_dict(torch.load(ckpt))
    m.eval()
    with torch.no_grad():
        # carried hidden state over the whole stream == running tick by tick
        out, h = [], None
        Xt = torch.tensor(X, device="cuda")
        for i in range(0, len(Xt), 1000):
            o, h = m(Xt[None, i:i + 1000], h)
            out.append(o[0].argmax(-1).cpu())
    del enc, m
    gc.collect()
    torch.cuda.empty_cache()
    return [tg.ACTIONS[i] for i in torch.cat(out).numpy()]


def score(answer, place):
    if place.split()[-1].lower() in answer.lower():
        return "correct"
    return "unknown" if UNKNOWN.search(answer) else "wrong"


def run(cond, ticks, acts, args):
    log = {"cond": cond, "questions": [], "writes": 0}
    mem = None
    if cond != "none":
        mem = MemoryModel(seed=args.seed)
        # the memory model has been alive before: adapt its senses to ordinary gists
        for x in world3.generate(3000, 5, "train"):
            if x["event"] and x["kind"] in ("greeting", "scene", "inform", "complex", "background"):
                mem.warm(mem.gist(to_memory_text(x["event"])))
                if mem.n >= 120:
                    break
    written_at = {}
    t0 = time.time()
    for i, (x, a) in enumerate(zip(ticks, acts)):
        ev = x["event"]
        if ev is None or ev.startswith("[THINKER"):
            continue
        if mem is not None and (cond == "all" or a != "WAIT"):
            g = mem.gist(to_memory_text(ev))
            mem.experience(g)
            log["writes"] += 1
            if x["kind"] == "inform":
                written_at[x["fact"][:2]] = log["writes"]
        if x["kind"] == "recall_q":
            nm, o, p = x["fact"]
            res = think(ev, mem.recall if mem else None)
            q = {"t": i, "q": ev, "truth": p, "model1": a, "answer": res["answer"], "label": score(res["answer"], p),
                 "recalls": res["trace"], "writes_since_fact": (log["writes"] - written_at[(nm, o)])
                 if (nm, o) in written_at else None}
            if a != "THINK":
                q["label_e2e"] = "missed"  # model 1 did not pass it to the thinker
            log["questions"].append(q)
            print(cond, i, q["label"], "|", ev, "->", res["answer"][:120], flush=True)
    log["seconds"] = round(time.time() - t0)
    del mem
    gc.collect()
    torch.cuda.empty_cache()
    return log


def main(args):
    ticks = world3.generate(args.ticks, seed=args.seed + 999, part="test", inform_rate=0.01, recall_rate=0.006)
    acts = model1_actions(ticks, args.ckpt)
    os.makedirs(args.out, exist_ok=True)
    for cond in args.conds.split(","):
        log = run(cond, ticks, acts, args)
        labs = [q["label"] for q in log["questions"]]
        log["summary"] = {k: labs.count(k) for k in ("correct", "unknown", "wrong")}
        print(cond, log["summary"], "writes", log["writes"], flush=True)
        json.dump(log, open(os.path.join(args.out, f"demo_{cond}_s{args.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conds", default="none,gated,all")
    p.add_argument("--ckpt", default=os.path.join(ROOT, "model1", "results_w3", "w3_qwen3emb_fam_s0.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "integration", "results"))
    main(p.parse_args())
