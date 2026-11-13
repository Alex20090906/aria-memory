"""Memory left by colloquial_test v2 (30 told facts): is a wrong inline recall a frame effect?
For each told fact whose gist was right, recall the place
  raw    : "Memory notes.\n[owner|obj] ->"            (the frame the value was written in)
  inline : chat-template frame with the colloquial question and the anchor prefilled in the assistant turn
"""
import json

import colloquial_test as ct
import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir

d = json.load(open("colloquial_result_v2.json", encoding="utf-8"))
facts = json.load(open("colloquial.json", encoding="utf-8"))["facts"][:30]
n_mem, D, k, n, _ = f.read_header()
E, P, st = fa.setup(D, k)
ok_raw = ok_inl = n_ok = 0
for w, x in zip([w for w in d["writes"] if not w.get("chatter")], facts):
    if not w["ok"]:
        continue
    o, obj, val = w["gist"]
    noun = val.split()[-1].lower()
    raw = fa.recall(f.PRE + f"[{o}|{obj}] ->", o, obj).strip().split("\n")[0]
    fa.off()
    q = f'{o} says: "{x["ask_casual"]}"'
    prompt = ir.render(ct.THINK_SYS, q, f"[{o}|{obj}] ->")
    toks, *_ = ir.capture_raw(prompt)
    anc = fa.anchor_units(E, P, st, toks, o, obj, k)
    fa.set_anchor(len(toks) - 1, anc, k, anchor_tok=int(toks[-1]))
    inl = ir.gen_raw(prompt, 12).strip().split("\n")[0]
    fa.off()
    n_ok += 1
    ok_raw += noun in raw.lower()
    ok_inl += noun in inl.lower()
    print(f"{o:6s} {obj:10s} truth={val[:22]!r:26s} raw={raw[:24]!r:28s} inline={inl[:24]!r}", flush=True)
print(f"RAW {ok_raw}/{n_ok}  INLINE {ok_inl}/{n_ok}")
