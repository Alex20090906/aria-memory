"""E40: does the write key equal the recall key? Tokenisation only, no memory write.

For each fact the write text "[owner|obj] -> value." and the recall text "[owner|obj] ->" are captured from the engine,
and the anchor units are built from both, with the old rule (last place each word occurs in the whole text) and the new
rule (inside the "[owner|obj]" key, FMEM_KEYFIX). A fact can only be recalled if its write and recall units overlap.
The first four facts are the E39 misses (the value repeats the object word); the rest are controls.

    FMEM_KEY=cross python keyfix_test.py
"""
import json
import sys

import fmem_anchor as fa
import fmem_tool as f

FACTS = [("Noah", "umbrella", "in the umbrella stand"), ("Carl", "pen", "in the pencil cup"),
         ("Emma", "medicine", "in the medicine cabinet"), ("Grace", "camera", "in the camera bag"),
         ("Vera", "watch", "on the nightstand"), ("Zack", "ticket", "in the coat pocket"),
         ("Lily", "keys", "under the doormat"), ("Owen", "phone", "on the bookshelf"),
         ("護照", "status", "過期且遺失，已重辦新護照"), ("備用眼鏡", "location", "書房防潮箱上層")]


def old_units(E, P, st, toks, owner, obj, k):
    ids_s = [int(toks[i]) for i in fa.token_ids_of(toks, owner)]
    ids_o = [int(toks[i]) for i in fa.token_ids_of(toks, obj)]
    if not ids_s or not ids_o:
        return None
    return fa.anchor_units_ids(E, P, st, ids_s, ids_o, k, owner, obj)


def main():
    n_mem, D, k, n, layers = f.read_header()
    E, P, st = fa.setup(D, k)
    rows = []
    for o, s, v in FACTS:
        tw, *_ = f.capture_request(f"[{o}|{s}] -> {v}.")
        tr, *_ = f.capture_request(f"[{o}|{s}] ->")
        rec = set(fa.anchor_units(E, P, st, tr, o, s, k).tolist())
        ow = old_units(E, P, st, tw, o, s, k)
        nw = fa.anchor_units(E, P, st, tw, o, s, k)
        rows.append({"fact": f"{o}|{s} -> {v}", "old_overlap": len(rec & set(ow.tolist())) if ow is not None else "empty",
                     "new_overlap": len(rec & set(nw.tolist())), "k": k})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    json.dump(rows, open("keyfix_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("E40 DONE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
