"""P8 offline, sound-aware finder (H25): many labels that the semantic finder misses are the same words written with
other characters by speech recognition (張副理 / 張富里, 修枝剪 / 修織剪, 檜木精油 / 繪墓精油). Embeddings compare
characters and meaning, so they miss these. This adds a second score: how much of the entity's sound (toneless pinyin,
easily confused initials and finals merged, as in the memory key's sound bank) matches.

Same replay and scoring as label_finder_embed.py. Finders compared at each threshold:
  E    semantic only (cosine >= tau)                       = what zh_v3 Z_FIND=embed does now
  P    sound only (sound score >= rho)
  E+P  sound match first (>= rho), otherwise semantic (>= tau)
An exact label is always answered directly; anything else is an ask-back.

    python label_finder_phon.py <zh_result_*.json> [...]        (P8_FM=<folder with timeline.json> for set2)
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_finder_eval as lf  # noqa: E402
import numpy as np  # noqa: E402
from label_finder_embed import INSTR, SLOT_ZH  # noqa: E402
from pypinyin import Style, lazy_pinyin  # noqa: E402

CONFUSE = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]


def syllables(text):
    out = []
    for s in lazy_pinyin(text.lower(), style=Style.NORMAL, errors=lambda x: list(x)):
        s = s.strip()
        if not s or not s.isalnum():
            continue
        for a, b in CONFUSE:
            s = s.replace(a, b)
        out.append(s)
    return out


def lcs(a, b):
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            m[i + 1][j + 1] = m[i][j] + 1 if a[i] == b[j] else max(m[i][j + 1], m[i + 1][j])
    return m[-1][-1]


def sound(a, b):
    """share of the shorter entity's syllables found in order in the other one (catches 託運單 in 鐵拖運單)"""
    x, y = syllables(a), syllables(b)
    if not x or not y:
        return 0.0
    if os.environ.get("P8_SOUND", "dice") == "dice":     # whole-entity agreement: 竹油 is not 保養豬油
        return 2 * lcs(x, y) / (len(x) + len(y))
    return lcs(x, y) / min(len(x), len(y)) if min(len(x), len(y)) >= 2 else float(x == y)


def run(path, enc, taus=(0.6,), rhos=(0.8, 0.9, 1.0)):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(lf.FM, "timeline.json"), encoding="utf-8"))}
    log = [{"label": [w["fact"][0], w["fact"][1]], "value": w["fact"][2], "day": w["day"], "start": w["start"],
            "src": w["src"]} for w in d["writes"]]
    labels = sorted({tuple(ev["label"]) for ev in log})
    K = enc.encode([f"{e}（{SLOT_ZH.get(s, s)}）" for e, s in labels], normalize_embeddings=True, batch_size=64)
    kidx = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for q in d["questions"]:
        qq = q["q"]
        seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        cand.sort(key=lambda a: -lf.overlap(qq["entity"], (a["anchor"] or "").split("|")[0]))
        a = cand[0] if cand else None
        told = qq["expect"] is not None
        if a is None or not a["anchor"]:
            rows.append({"told": told, "routed": False})
            continue
        e, s = (a["anchor"].strip("[]").split("|") + [""])[:2]
        before = [ev for ev in log if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                               and ev["start"] < seg["start"])]
        exact = lf.find("F0", e, s, qq["text"], before)
        latest = {}
        for ev in before:
            latest[tuple(ev["label"])] = ev["value"]
        emb = ph = None
        if latest:
            labs = list(latest)
            qv = enc.encode(INSTR + f"{qq['text']}（{e}，{SLOT_ZH.get(s, s)}）", normalize_embeddings=True)
            sims = K[[kidx[l] for l in labs]] @ qv
            j = int(np.argmax(sims))
            emb = (labs[j], latest[labs[j]], float(sims[j]))
            # sound: same entity sound, the asked slot first, then the latest entry
            sc = [(sound(e, l[0]), l[1] == s, i) for i, l in enumerate(labs)]
            j = max(range(len(labs)), key=lambda i: sc[i])
            ph = (labs[j], latest[labs[j]], sc[j][0])
        rows.append({"told": told, "routed": True, "expect": qq["expect"], "entity": qq["entity"],
                     "anchor": f"[{e}|{s}]", "exact": exact, "emb": emb, "ph": ph})
    print("==", os.path.basename(path), flush=True)

    def pick(r, mode, tau, rho):
        if mode in ("P", "E+P") and r["ph"] and r["ph"][2] >= rho:
            return r["ph"]
        if mode in ("E", "E+P") and r["emb"] and r["emb"][2] >= tau:
            return r["emb"]
        return None

    for tau in taus:
        for mode, rr in (("E", (9,)), ("P", rhos), ("E+P", rhos)):
            for rho in rr:
                c = dict(direct_ok=0, direct_wrong=0, ask_ok=0, ask_wrong=0, none=0, not_routed=0, never_direct=0,
                         never_ask=0)
                for r in rows:
                    if not r["routed"]:
                        c["not_routed"] += r["told"]
                        continue
                    if r["exact"] is not None:
                        if r["told"]:
                            c["direct_ok" if lf.overlap(r["expect"], r["exact"][0]) >= 0.6 else "direct_wrong"] += 1
                        else:
                            c["never_direct"] += 1
                        continue
                    b = pick(r, mode, tau, rho)
                    if b is None:
                        c["none"] += r["told"]
                    elif r["told"]:
                        c["ask_ok" if lf.overlap(r["expect"], b[1]) >= 0.6 else "ask_wrong"] += 1
                    else:
                        c["never_ask"] += 1
                name = mode if mode == "E" else f"{mode} rho {rho}"
                print(f"  tau {tau} {name:12s}: right {c['direct_ok'] + c['ask_ok']:2d} (direct {c['direct_ok']}, "
                      f"ask {c['ask_ok']}) | ask wrong {c['ask_wrong']} | direct wrong {c['direct_wrong']} | none "
                      f"{c['none']} | never asked {c['never_ask']} answered {c['never_direct']}", flush=True)
    for r in rows:
        if r["routed"] and r["exact"] is None and r["ph"] and r["ph"][2] >= 0.75:
            b = r["ph"]
            ok = (lf.overlap(r["expect"], b[1]) >= 0.6) if r["told"] else None
            print(f"    sound {'told ' if r['told'] else 'never'} {r['entity']} {r['anchor']} -> {'|'.join(b[0])} "
                  f"{b[2]:.2f} {b[1][:20]!r} ok={ok}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    for p in sys.argv[1:]:
        run(p, enc)
