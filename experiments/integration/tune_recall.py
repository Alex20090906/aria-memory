"""Tune fast-weight writing for canonical (gist) facts: immediate vs after 20 facts."""
import random, sys, itertools
sys.path.insert(0, "experiments/model1"); sys.path.insert(0, "experiments/integration")
import torch, world3
from memory_v1 import MemoryModel
mem = MemoryModel()
warm = [x["event"].split("] ", 1)[1] for x in world3.generate(3000, 5, "train") if x["event"] and x["kind"] != "empty"][:150]
B0 = None
for steps, lr, k in [(5, 1e-2, 32), (10, 1e-2, 32), (5, 3e-2, 32), (10, 1e-2, 64), (10, 3e-2, 16)]:
    rng = random.Random(0)
    mem.__init__(k=k, lr=lr, steps=steps) if False else None
    mem.B.data.zero_(); mem.opt = torch.optim.Adam([mem.B], lr=lr); mem.steps = steps; mem.k = k
    mem.codes, mem.strength, mem.mu, mem.tmu, mem.n = [], [], None, None, 0
    for w in warm: mem.warm(w)
    facts, imm = [], []
    for o in rng.sample(world3.OBJS, 20):
        nm, p = rng.choice(world3.world2.NAMES), rng.choice(world3.PLACES).replace("my ", "")
        fact = f"{nm}'s {o} is {p}."
        mem.experience(fact)
        imm.append(p.split()[-1] in mem.recall(f"{nm}'s {o} is")["recalled"])
        facts.append((nm, o, p))
    late = [p.split()[-1] in mem.recall(f"{nm}'s {o} is")["recalled"] for nm, o, p in facts]
    print(f"steps {steps} lr {lr} k {k}: immediate {sum(imm)/20:.2f} after-20 {sum(late)/20:.2f}", flush=True)
