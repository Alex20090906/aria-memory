import inline_recall as ir, fmem_anchor as fa, fmem_tool as f, random
n_mem, D, k, n, layers = f.read_header(); E, P, st = fa.setup(D, k)
f.reset(None); fa.write("Leo", "passport", "in the freezer")
rng = random.Random(7)
filler = ["The kitchen is warm.", "Rain again today.", "Someone left the light on.", "Dinner was late.", "The cat is asleep."]
for trial in range(4):
    pre = " ".join(rng.choice(filler) for _ in range(rng.randint(0, 20)))
    prompt = f.PRE + pre + "\nWhere is Leo's passport?\n[Leo|passport] ->"
    fa.off(); toks, resume, rows, _ = ir.capture_raw(prompt)
    anc = fa.anchor_units(E, P, st, toks, "Leo", "passport", k)
    out = {}
    for off in [0, -4]:
        fa.set_anchor(len(toks) - 1 + off, anc, k)
        out[off] = ir.gen_raw(prompt, 4)
    # now a *fresh* variant (never captured) with anchor set from a capture of a different prompt length? no: same prompt, second gen
    fa.set_anchor(len(toks) - 1, anc, k); again0 = ir.gen_raw(prompt, 4)
    print("ntok", len(toks), "resume", resume, "rows", {s: r.shape for s, r in rows.items()}, "| off0", repr(out[0]), "| off-4", repr(out[-4]), "| off0 again", repr(again0))
fa.off()
