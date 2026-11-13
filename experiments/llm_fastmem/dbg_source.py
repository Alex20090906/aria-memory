"""Source monitoring on the memory left by colloquial_test v3: complete each anchor with the memory ON and OFF.
If the ON completion equals the OFF completion, the words come from the model's prior, not from memory.
Reports, for told facts (gist right) and for never-told facts, how often ON is correct and how often ON == OFF."""
import json

import colloquial_test as ct
import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir

d = json.load(open("colloquial_result_v3.json", encoding="utf-8"))
facts = json.load(open("colloquial.json", encoding="utf-8"))["facts"]
n_mem, D, k, n, _ = f.read_header()
E, P, st = fa.setup(D, k)


def first_words(s, n=3):
    return " ".join(s.lower().replace(".", " ").split()[:n])


rows = []
told = [w for w in d["writes"] if not w.get("chatter")]
for i, x in enumerate(facts[:40]):
    w = told[i] if i < 30 else None
    o, obj = x["owner"], x["object"]
    q = f'{o} says: "{x["ask_casual"]}"'
    prompt = ir.render(ir.THINK_SYS, q, f"[{o}|{obj}] ->")
    toks, *_ = ir.capture_raw(prompt)
    fa.off()
    off = ir.gen_raw(prompt, 10).strip().split("\n")[0]
    anc = fa.anchor_units(E, P, st, toks, o, obj, k)
    fam = fa.familiarity(anc, D)
    fa.set_anchor(len(toks) - 1, anc, k, anchor_tok=int(toks[-1]))
    on = ir.gen_raw(prompt, 10).strip().split("\n")[0]
    fa.off()
    noun = x["place"].split()[-1].lower()
    kind = "never" if i >= 30 else ("told" if w["ok"] else "gist_failed")
    same = first_words(on) == first_words(off)
    rows.append({"kind": kind, "on_ok": noun in on.lower(), "same": same, "theta": fam})
    print(f"{kind:11s} theta={fam:.2f} same={same!s:5s} ok={noun in on.lower()!s:5s} on={on[:28]!r:32s} off={off[:28]!r}",
          flush=True)
for kind in ("told", "gist_failed", "never"):
    r = [x for x in rows if x["kind"] == kind]
    if not r:
        continue
    print(kind, "n", len(r), "on_ok", sum(x["on_ok"] for x in r), "same_as_prior", sum(x["same"] for x in r),
          "on_ok_and_not_same", sum(x["on_ok"] and not x["same"] for x in r),
          "wrong_and_not_same", sum((not x["on_ok"]) and not x["same"] for x in r))
