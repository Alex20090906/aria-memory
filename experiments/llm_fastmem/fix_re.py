import re, sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().split("\n")
for i, l in enumerate(lines):
    if l.startswith("PLACE_RE"):
        lines[i] = r'''PLACE_RE = re.compile(r"^(in|on|under|behind|inside|at|next to|near|beside|on top of) [A-Za-z][\w ,'-]{2,40}$", re.I)'''
s = "\n".join(lines)
open(p, "w", encoding="utf-8").write(s)
ns = {"re": re}
exec([l for l in lines if l.startswith("PLACE_RE")][0], ns)
for v in ['in the blue envelope', 'on top of the fridge', '100', '1234567', '2025-12', 'in the flask, there is a liquid', 'under the doormat']:
    print(repr(v), bool(ns["PLACE_RE"].match(v)))
