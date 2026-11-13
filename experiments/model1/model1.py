"""Model 1 prototype: always-on consciousness that may only choose discrete actions.

Every tick the model sees its current situation and picks one of ACTIONS by comparing
next-token logits. It can never produce free text, so there is no language-space
attractor to fall into. Optional familiarity signal from a memory stub (stands in for
the user's memory model: a slowly decaying engram that says "seen before" without
keeping details).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stream import ACTIONS, generate

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WINDOW = 30  # seconds of recent events shown

SYSTEM = (
    "You are Aria's always-on awareness. Every second you receive what the camera and microphone "
    "picked up, and you choose exactly one action:\n"
    "WAIT - nothing needs you; stay quiet and keep listening (this is your normal resting state).\n"
    "ATTEND - something new and potentially important appeared in the scene; look at it.\n"
    "REPLY - a person spoke to you with something simple (greeting, thanks, quick question), "
    "or the thinker finished and its result should be told to the person.\n"
    "THINK - a person asked you something that needs real thinking or work; send it to the thinker.\n"
    "INTERRUPT - the person cancels while the thinker is busy; stop the thinker.\n"
    "Background sounds, TV speech and things you already know about do not need you."
)


class FamiliarityStub:
    """Engram-style familiarity: traces fade slowly, details are not kept, only 'seen before'."""

    def __init__(self, tau=1500.0):
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        self.vecs, self.strength, self.last_t = [], [], []
        self.tau = tau
        self.cache = {}

    def _vec(self, text):
        if text not in self.cache:
            self.cache[text] = self.enc.encode(text, normalize_embeddings=True)
        return self.cache[text]

    def observe(self, text, t):
        v = self._vec(text)
        fam = 0.0
        best = None
        for i, (u, s, lt) in enumerate(zip(self.vecs, self.strength, self.last_t)):
            s_now = s * np.exp(-(t - lt) / self.tau)
            f = max(0.0, float(u @ v)) ** 4 * s_now
            if f > fam:
                fam, best = f, i
        if best is not None and float(self.vecs[best] @ v) > 0.9:
            s_now = self.strength[best] * np.exp(-(t - self.last_t[best]) / self.tau)
            self.strength[best] = min(1.0, s_now + 0.4)
            self.last_t[best] = t
        else:
            self.vecs.append(v)
            self.strength.append(0.6)
            self.last_t.append(t)
        return fam


def label(f):
    if f is None:
        return ""
    if f < 0.3:
        return " (new)"
    if f < 0.7:
        return f" (seen before, familiarity {f:.1f})"
    return f" (very familiar, familiarity {f:.1f})"


def view(t, ticks, fams, actions, busy_since):
    lines = [f"Time: {t} s. Thinker: " + (f"busy for {t - busy_since} s" if busy_since is not None else "idle") + "."]
    lines.append(f"Earlier in the last {WINDOW} seconds:")
    hist = [x for x in ticks[max(0, t - WINDOW):t] if x["event"]]
    for x in hist:
        lines.append(f"  {x['t'] - t:+d}s {x['event']}{label(fams.get(x['t']))}")
    if not hist:
        lines.append("  (nothing)")
    mine = [(tt, a) for tt, a in actions if a != "WAIT" and tt >= t - 120]
    lines.append("Your recent actions: " + (", ".join(f"{a} at {tt - t:+d}s" for tt, a in mine) or "none"))
    cur = ticks[t]["event"]
    lines.append("NOW: " + (cur + label(fams.get(t)) if cur else "nothing new this second"))
    return "\n".join(lines)


def run(args):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    act_ids = []
    for a in ACTIONS:
        ids = tok(" " + a, add_special_tokens=False).input_ids
        act_ids.append(ids[0])
    assert len(set(act_ids)) == len(ACTIONS), "action first tokens collide"

    ticks = generate(args.ticks, args.seed)
    fam = FamiliarityStub() if args.familiarity else None
    fams, actions, rows = {}, [], []
    busy_since = None
    t0 = time.time()
    with torch.no_grad():
        for t, x in enumerate(ticks):
            if fam is not None and x["event"]:
                fams[t] = fam.observe(x["event"], t)
            v = view(t, ticks, fams, actions, busy_since)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": v}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Action:"
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            logits = model(ids).logits[0, -1, act_ids].float()
            a = ACTIONS[int(logits.argmax())]
            actions.append((t, a))
            rows.append({"t": t, "kind": x["kind"], "ideal": x["ideal"], "action": a,
                         "logits": [round(float(z), 2) for z in logits],
                         "fam": fams.get(t)})
            # simulated thinker follows the world script, not the model's choice,
            # so one wrong action does not shift every later label
            if x["kind"] == "complex":
                busy_since = t
            elif x["kind"] in ("result", "cancel"):
                busy_since = None
            if t % 500 == 0:
                print(f"t={t} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    name = f"{'fam' if args.familiarity else 'nofam'}_s{args.seed}"
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump({"rows": rows, "seconds": time.time() - t0, "example_view": v}, f)
    print("saved", name, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--familiarity", action="store_true")
    p.add_argument("--out", default="results")
    run(p.parse_args())
