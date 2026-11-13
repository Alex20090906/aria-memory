"""World v3 = world v2 plus two kinds needed for the three-model integration:

inform   : a person tells Aria a personal fact ("Aria, I put my passport in the blue box")
           -> REPLY (short acknowledgement). These are the events worth remembering.
recall_q : a person asks Aria about something they told her earlier
           ("Aria, where did I put my passport?") -> THINK (needs memory + thinking).
"""
import random

import world2
from world2 import ACTIONS, THINK_TICKS, split  # noqa: F401

OBJS = ["passport", "keys", "wallet", "glasses", "charger", "umbrella", "headphones", "notebook", "watch", "medicine",
        "ticket", "scarf", "camera", "ring", "remote control", "spare key", "bank card", "laptop", "water bottle", "gloves"]
PLACES = ["in the blue box", "under the bed", "on the bookshelf", "in the car", "in the kitchen drawer",
          "behind the sofa", "on the windowsill", "in the closet", "in the garage", "next to the TV",
          "in my backpack", "on the desk", "in the bathroom cabinet", "under the doormat"]
INFORM_FORMS = ["Aria, I put my {o} {p}.", "Aria, just so you know, I left my {o} {p}.", "Aria, my {o} is {p} now.",
                "Aria, remember that I put the {o} {p}.", "Aria, I'm leaving my {o} {p}.", "Aria, FYI the {o} is {p}.",
                "Aria, I've stored my {o} {p}.", "Aria, note that my {o} went {p}."]
RECALL_FORMS = ["Aria, where did I put my {o}?", "Aria, do you remember where my {o} is?",
                "Aria, I can't find my {o}, where did I leave it?", "Aria, where is my {o}?",
                "Aria, help me remember where I put the {o}.", "Aria, any idea where my {o} went?"]


def banks(part):
    b = world2.banks(part)
    rng = random.Random(4321)
    for name, xs in [("objs", OBJS), ("inform", INFORM_FORMS), ("recall", RECALL_FORMS)]:
        tr, te = split(xs, rng)
        b[name] = tr if part == "train" else te
    b["places"] = PLACES
    return b


def generate(n_ticks, seed, part, inform_rate=0.008, recall_rate=0.004):
    b = banks(part)
    rng = random.Random(seed + 77)
    ticks = world2.generate(n_ticks, seed, part)
    told = []
    for x in ticks:
        if x["event"] is not None or x["busy"]:
            continue
        r = rng.random()
        nm = rng.choice(b["names"])
        if r < inform_rate:
            o, p = rng.choice(b["objs"]), rng.choice(b["places"])
            x.update(event=f"[MIC] {nm}: " + rng.choice(b["inform"]).format(o=o, p=p), ideal="REPLY", kind="inform",
                     fact=(nm, o, p))
            told.append((nm, o, p))
        elif r < inform_rate + recall_rate and told:
            # only free ticks that are not inside a thinker job, so the busy schedule stays valid
            # the person who told the fact asks about it; the truth is the latest place they gave
            nm, o, _ = rng.choice(told)
            p = [f for f in told if f[0] == nm and f[1] == o][-1][2]
            x.update(event=f"[MIC] {nm}: " + rng.choice(b["recall"]).format(o=o), ideal="THINK",
                     kind="recall_q", fact=(nm, o, p))
    return ticks
