"""Average world2 results over seeds, one row per (encoder, fam) condition."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results_w2")
groups = defaultdict(list)
for f in glob.glob(os.path.join(d, "w2_*.json")):
    name = os.path.basename(f).rsplit("_s", 1)[0]
    groups[name].append(json.load(open(f)))
keys = ["ambient", "background", "overheard", "greeting", "complex", "cancel", "result", "scene", "scene_return", "scene_repeat"]
print("cond".ljust(22), "n", "falseQ".rjust(7), *[k[:9].rjust(9) for k in keys], "Q1/Q4".rjust(11))
for name, rs in sorted(groups.items()):
    m = lambda k: np.mean([r["per_kind"].get(k, np.nan) for r in rs])
    print(name.ljust(22), len(rs), f"{np.mean([r['false_action_quiet'] for r in rs])*100:6.2f}%",
          *[f"{m(k):9.3f}" for k in keys],
          f"{np.mean([r['first_quarter'] for r in rs]):.2f}/{np.mean([r['last_quarter'] for r in rs]):.2f}".rjust(11))
for name, rs in sorted(groups.items()):
    print(name, "confusions s0:", rs[0]["confusions"][:5])
