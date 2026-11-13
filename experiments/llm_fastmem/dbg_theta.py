"""Why is [護照|location] unfamiliar right after it was written? Compare keys from the three paths."""
import numpy as np

import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir
from zh_memory_test import THINK_SYS

n_mem, D, k, n, _ = f.read_header()
E, P, st = fa.setup(D, k)
th = fa.theta_load(D)
print("KEY", fa.KEY, "theta>0.5 units:", int((th > 0.5).sum()), "max", float(th.max()))
for e, s in [("護照", "location"), ("備用鑰匙", "location"), ("體重計", "location")]:
    toks, *_ = f.capture_request(f"[{e}|{s}] ->")
    a1 = fa.anchor_units(E, P, st, toks, e, s, k)
    a2 = fa.units_batch(E, P, st, [(e, s)], k)[0]
    prompt = ir.render(THINK_SYS, f'使用者說：「{e}咧」', f"[{e}|{s}] ->")
    t3, *_ = ir.capture_raw(prompt)
    a3 = fa.anchor_units(E, P, st, t3, e, s, k)
    print(e, s, "write-path", fa.familiarity(a1, D), "batch", fa.familiarity(a2, D) if a2 is not None else None,
          "think-path", fa.familiarity(a3, D), "| same w/b", set(a1) == set(a2 if a2 is not None else []),
          "same w/t", set(a1) == set(a3), "theta per unit (write path)", np.round(th[a1], 2).tolist())
