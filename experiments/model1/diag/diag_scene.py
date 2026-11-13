"""Split model 1 errors on novel scenes: first-ever sighting vs return after the habituation window."""
import sys, json
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from world import ACTIONS, generate
from train_gru import featurize, Model1, EMB
from sentence_transformers import SentenceTransformer
seed = 0
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
ticks = generate(30000, seed=seed + 999, part="test")
X, y = featurize(ticks, {}, enc, seed + 1)
m = Model1(X.shape[1]).cuda(); m.load_state_dict(torch.load(sys.argv[2])); m.eval()
with torch.no_grad():
    p = m(torch.tensor(X, device="cuda")[None])[0][0].argmax(-1).cpu().numpy()
first, seen = {}, {}
rows = defaultdict(list)
for i, x in enumerate(ticks):
    if x["kind"] in ("scene", "scene_repeat"):
        s = x["event"]
        gap = i - seen[s] if s in seen else None
        seen[s] = i
        if x["kind"] == "scene":
            key = "first" if gap is None else ("gap400-800" if gap < 800 else "gap800+")
            rows[key].append((ACTIONS[p[i]] == "ATTEND", X[i, EMB + 1], X[i, EMB + 4]))
        else:
            rows["repeat<400"].append((ACTIONS[p[i]] == "WAIT", X[i, EMB + 1], X[i, EMB + 4]))
for k, v in rows.items():
    v = np.array(v)
    print(k, "n", len(v), "acc", v[:, 0].mean().round(3), "fam_slow", v[:, 1].mean().round(3), "fam_fast", v[:, 2].mean().round(3))
