"""Anchor-unit overlap across the exp_c fact set (no engine needed): how many facts share each unit,
and how the first answer token's value collides (all first tokens are written on the anchor units themselves)."""
import random, collections, numpy as np
import fmem_anchor as fa, fmem_tool as f
from exp_c import NAMES, OBJS, PLACES
n_mem, D, k, n, layers = f.read_header(); E, P, st = fa.setup(D, k)
vb = f.id_to_bytes(); inv = {v: i for i, v in vb.items()}
def ids(word):
    out = []
    for w in word.split(" "):
        b = b" " + w.encode()
        while b:                      # greedy longest-prefix match in the byte vocab
            for L in range(len(b), 0, -1):
                if b[:L] in inv:
                    out.append(inv[b[:L]]); b = b[L:]; break
            else:
                raise KeyError(b)
    return out
P32 = np.asarray(P, np.float32)
def emb(idl):
    x = E[idl].astype(np.float32)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)
    return (x.mean(0) - st["mu"]) / st["sd"]
def units(ia, io):
    s = (emb(ia) @ P32) * np.roll(emb(io) @ P32, 1)
    return np.sort(np.argpartition(-s, k)[:k]).astype(np.int32)
print("Leo", ids("Leo"), "Yumi", ids("Yumi"), "spare key", ids("spare key"))
for nfacts in (40, 100, 200):
    rng = random.Random(0); pairs = [(a, o) for a in NAMES for o in OBJS]; rng.shuffle(pairs)
    use = collections.Counter(); keys = []
    for a, o in pairs[:nfacts]:
        anc = units(ids(a), ids(o)); keys.append(set(anc.tolist())); use.update(anc.tolist())
    shared = sum(1 for u, c in use.items() if c > 1)
    jac = [len(keys[i] & keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    print(f"n={nfacts}: units used {len(use)} of {D}, shared by >1 fact {shared}, max facts per unit {max(use.values())}, "
          f"pairs with >=4 common units {sum(j >= 4 for j in jac)}, mean common {np.mean(jac):.2f}")
