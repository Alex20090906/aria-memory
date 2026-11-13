import os, re
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleep.py")
s = open(p, encoding="utf-8").read()
# remove the broken "protect recent" block (from its comment to the end of the for loop) and replace it cleanly
start = s.index("        # events of the last wake window")
end = s.index("        B[mask] = (B[mask]")
new = '''        # events of the last wake window are still "unjudged" (not yet asked): keep their rows intact but promote
        # nothing. Their answer rows depend on the stored tokens, which we do not keep, so they are re-derived from a
        # gated recall of each fact (the hippocampal index re-cues the cortex).
        for ids_s, ids_o in self.ring[-T_SLEEP:]:
            owner, obj = self.text(ids_s), self.text(ids_o)
            anc, _ = self.units_of(owner, obj)
            mask[anc] = False
            out, _ = self.recall(owner, obj, bump=False)
            val = out.split("\\n")[0].rstrip(".").strip()
            if val:
                text = f"[{owner}|{obj}] -> {val}."
                t2, *_ = f.capture_request(text)
                pos = fa.token_ids_of(t2, text)
                a_end = max(i for i in pos if b"->" in self.vb.get(int(t2[i]), b""))
                for t in range(a_end + 1, pos[-1]):
                    mask[fa.shifted(anc, t - a_end, t2[t], self.D)] = False
'''
s = s[:start] + new + s[end:]
# replay from a memory decayed by e^(-41/40) already fails (first run: facts written >30 events before a sleep were
# consolidated as garbage); fast tau 100 keeps the worst case at e^(-50/100) = 0.61
s = s.replace("TAU_FAST, TAU_SLOW, T_SLEEP, DOWNSCALE = 40.0, 500.0, 50, 0.1",
              "TAU_FAST, TAU_SLOW, T_SLEEP, DOWNSCALE = 100.0, 500.0, 50, 0.1")
open(p, "w", encoding="utf-8").write(s)
import ast; ast.parse(s); print("ok")
