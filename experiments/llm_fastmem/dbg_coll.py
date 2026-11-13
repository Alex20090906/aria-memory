import json
import inline_recall as ir, colloquial_test as ct, fmem_anchor as fa
fa.off()
d = json.load(open("colloquial.json", encoding="utf-8"))["facts"]
FEW = ct.THINK_SYS + ("\n\nExamples of how you start:\n"
    'Ken says: "ugh where did I leave my charger again??" -> you write: [Ken|charger] ->\n'
    'Mia says: "Aria, my wallet?" -> you write: [Mia|wallet] ->\n'
    'Tom says: "did I ever tell you where my ring is?" -> you write: [Tom|ring] ->\n'
    'Sara says: "any idea where Leo put his scarf? he is freaking out" -> you write: [Leo|scarf] ->\n'
    "Always start your reply with the anchor line.")
for name, sysm in [("plain", ct.THINK_SYS), ("few", FEW)]:
    ok = 0; n = 0; bad = []
    for x in d[:6]:
        for kind in ("ask_casual", "ask_terse", "ask_meta", "ask_other"):
            sp = x["other"] if kind == "ask_other" else x["owner"]
            head = ir.gen_raw(ir.render(sysm, f'{sp} says: "{x[kind]}"'), 60, stop=["->"]) or ""
            m = ir.ANCHOR_RE.search(head.rstrip() + (" ->" if not head.rstrip().endswith("->") else ""))
            good = bool(m) and m.group(1).strip().lower() == x["owner"].lower() and x["object"].split()[-1] in m.group(2).lower()
            ok += good; n += 1
            if not good: bad.append(head[:60])
    print(name, ok, "/", n, bad[:4], flush=True)
