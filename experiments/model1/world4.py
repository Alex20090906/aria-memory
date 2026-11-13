"""World v4 = world v3 plus a separate ENCODE label (fix for "observation amnesia", raised in the review).

Acting and remembering are different decisions. Model 1 now outputs both:
  action  WAIT / ATTEND / REPLY / THINK / INTERRUPT   (should I do something?)
  encode  0 / 1                                       (is this worth writing to memory?)

New kind:
  overheard_fact  people talk to each other and a fact is said ("Mia to Ken: I left my keys in the car")
                  -> action WAIT (do not butt in) but encode 1 (remember it)
Encode labels: 1 for facts (inform, overheard_fact), requests and questions (complex, recall_q, overheard),
new scenes and scene returns; 0 for ambient noise, TV/radio, repeated scenes, greetings, cancels, results.
"""
import random

import world3
from world2 import ACTIONS, split  # noqa: F401

ENCODE_KINDS = {"inform", "overheard_fact", "complex", "recall_q", "overheard", "scene", "scene_return"}
OVERHEARD_FACT_FORMS = ["{a} to {b}: I left my {o} {p}.", "{a} tells {b}: my {o} is {p}.",
                        "{a} says to {b}: I put the {o} {p}.", "{a} to {b}: by the way, my {o} is {p}.",
                        "{a} on the phone: I think my {o} is {p}.", "{a} to {b}: remind me, the {o} is {p}."]
RECALL_OTHER_FORMS = ["Aria, do you know where {a} left the {o}?", "Aria, where did {a} put the {o}?",
                      "Aria, any idea where {a}'s {o} is?", "Aria, did {a} say where the {o} is?"]


def banks(part):
    b = world3.banks(part)
    rng = random.Random(999)
    for name, xs in [("ofact", OVERHEARD_FACT_FORMS), ("recall_other", RECALL_OTHER_FORMS)]:
        tr, te = split(xs, rng)
        b[name] = tr if part == "train" else te
    return b


def generate(n_ticks, seed, part, inform_rate=0.008, recall_rate=0.004, ofact_rate=0.006):
    b = banks(part)
    rng = random.Random(seed + 4242)
    ticks = world3.generate(n_ticks, seed, part, inform_rate, recall_rate)
    heard = []
    for x in ticks:
        if x["event"] is None and not x["busy"]:
            r = rng.random()
            if r < ofact_rate:
                a, c = rng.sample(b["names"], 2)
                o, p = rng.choice(b["objs"]), rng.choice(b["places"])
                x.update(event="[MIC] " + rng.choice(b["ofact"]).format(a=a, b=c, o=o, p=p), ideal="WAIT",
                         kind="overheard_fact", fact=(a, o, p))
                heard.append((a, o, p))
            elif r < ofact_rate * 1.6 and heard:
                a, o, _ = rng.choice(heard)
                p = [f for f in heard if f[0] == a and f[1] == o][-1][2]
                asker = rng.choice([n for n in b["names"] if n != a])
                x.update(event=f"[MIC] {asker}: " + rng.choice(b["recall_other"]).format(a=a, o=o), ideal="THINK",
                         kind="recall_q_other", fact=(a, o, p))
        x["encode"] = int(x["kind"] in ENCODE_KINDS or x["kind"] == "recall_q_other")
    return ticks


if __name__ == "__main__":
    from collections import Counter
    t = generate(30000, 0, "test")
    print(Counter(x["kind"] for x in t))
    print(Counter((x["ideal"], x["encode"]) for x in t if x["event"]))
