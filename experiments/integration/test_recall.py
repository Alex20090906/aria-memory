"""How robust is pattern-completion recall to the cue wording? Store 20 facts said in varied
forms, then cue with (A) the exact stored prefix, (B) 'X said: my OBJ is', (C) 'X said: I put my OBJ'."""
import json, random, sys
sys.path.insert(0, "experiments/model1"); sys.path.insert(0, "experiments/integration")
import world3
from memory_v1 import MemoryModel
layer = int(sys.argv[1]) if len(sys.argv) > 1 else 14
GIST = len(sys.argv) > 2 and sys.argv[2] == 'gist'
rng = random.Random(0)
mem = MemoryModel(layer=layer)
# the memory model has been 'alive' before the test: adapt to ordinary events first
for x in world3.generate(3000, 5, "train"):
    if x["event"] and x["kind"] in ("ambient", "background", "greeting", "scene", "overheard", "complex"):
        mem.warm(x["event"].split("] ", 1)[1])
        if mem.n >= 150: break
facts = []
objs = rng.sample(world3.OBJS, 20)
for o in objs:
    nm, p = rng.choice(world3.world2.NAMES), rng.choice(world3.PLACES)
    form = rng.choice(world3.INFORM_FORMS)
    utt = form.format(o=o, p=p).replace("Aria, ", "")
    text = f"{nm} said: {utt}"
    g = mem.gist(text) if GIST else text
    print(text, '=>', g)
    mem.experience(g)
    facts.append((nm, o, p, text))
F = (lambda nm, o, p: f"{nm}'s {o} is {p}.") if GIST else (lambda nm, o, p: f"{nm} said: my {o} is {p}.")
res = {"A": [], "B": [], "C": []}
fam = []
for nm, o, p, text in facts:
    pre = text[:text.index(o) + len(o)]
    for key, cue in [("A", f"{nm}'s {o} is" if GIST else pre), ("B", f"{nm} said: my {o} is"), ("C", f"{nm} said: I put my {o}")]:
        r = mem.recall(cue)
        res[key].append(p.split()[-1] in r["recalled"])
        if key == "B":
            lure_p = rng.choice([q for q in world3.PLACES if q != p])
            fam.append((r["cue_familiarity"], mem.familiarity(F(nm, o, p)),
                        mem.familiarity(F(nm, o, lure_p)),
                        mem.familiarity(F(nm, "lamp", p)), r["recalled"]))
print("layer", layer, {k: sum(v) / len(v) for k, v in res.items()})
for f in fam[:8]:
    print("cue %.2f true %.2f lure %.2f unrelated-obj %.2f | %s" % f)
