"""Colloquial test on LLM server: gist (LLM server) -> in-weights write -> inline recall with casual questions.

colloquial.json (from experiments/colloquial/gen_colloquial.py): facts with tell_direct / tell_indirect /
ask_casual / ask_terse / ask_meta / ask_other, plus chatter lines.
Schedule: facts 0..29 are told (even: direct, odd: indirect), facts 30..39 are never told (negatives);
chatter lines are interleaved and also go through the gist (false writes are counted).
Then every fact is asked four ways. Scores:
  gist     told facts whose gist gives the right owner/object/place (place main noun present)
  recall   told facts: answer contains the place's main noun and no "don't remember"
  honest   never-told facts: answer says it does not remember (no place from the place list)
Owner/object are canonicalised (Title-case owner, lowercase object without articles/possessives) at write AND at
recall, so "[Leo|Passport]" and "[Leo|passport]" hit the same key.
"""
import json
import os
import re
import sys

import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir

HERE = os.path.dirname(os.path.abspath(__file__))
GIST_SYS = ("Extract a location fact from what a person said near the home assistant Aria. Output exactly one line "
            "'owner | object | place' when the utterance states or clearly implies where someone's thing is, using the "
            "speaker's name for 'my'/'I', the bare object noun, and the place as a short phrase starting with a "
            "preposition (in/on/under/behind/...). Otherwise output NONE (questions, chatter, plans, jokes without a "
            "location). No other text.")
GIST_SHOTS = [("Mia: Aria, just so you know, I left my camera on the desk.", "Mia | camera | on the desk"),
              ("Tom: don't run the dishwasher, my ring's sitting in the cutlery basket lol", "Tom | ring | in the cutlery basket"),
              ("Leo: hey where'd my passport end up again", "NONE"),
              ("Sara: ugh this rain is never gonna stop", "NONE"),
              ("Ken: Aria did I ever tell you where my charger is?", "NONE")]
RESOLVE = os.environ.get("COLL_RESOLVE", "1") == "1"   # anchor decided by resolve_query (v3) or written freely (v2)
THINK_SYS = ("You are Aria's thinking mind. The line you get starts with the speaker's name. You have no notes. To "
             "remember where a person's thing is, write the anchor [owner|object] -> using the owner's name and the bare "
             "object noun, and the place comes to you right after the arrow. If nothing comes after the arrow, say you do "
             "not remember. The words after the arrow are your own memory: trust them and use them.\n\n"
             "Examples of how you start:\n"
             'Ken says: "ugh where did I leave my charger again??" -> you write: [Ken|charger] ->\n'
             'Mia says: "Aria, my wallet?" -> you write: [Mia|wallet] ->\n'
             'Tom says: "did I ever tell you where my ring is?" -> you write: [Tom|ring] ->\n'
             'Sara says: "any idea where Leo put his scarf? he is freaking out" -> you write: [Leo|scarf] ->\n'
             "Always start your reply with the anchor line.")
UNKNOWN = re.compile(r"(don[’']t|do not|can[’']t|cannot|couldn[’']t|could not|no idea|not sure)\b", re.I)
ART = re.compile(r"^(my|his|her|their|the|a|an|your|our)\s+", re.I)


def canon(owner, obj):
    owner = owner.strip().split()[0].strip("'’s").capitalize() if owner.strip() else owner
    obj = obj.strip().lower().rstrip(".")
    while ART.match(obj):
        obj = ART.sub("", obj)
    return owner, obj


def gist(line):
    msgs = [{"role": "system", "content": GIST_SYS}]
    for a, b in GIST_SHOTS:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": line})
    fa.off()
    out = (f.chat(msgs, max_tokens=30) or "").strip().splitlines()
    out = out[0] if out else ""
    if out.upper().startswith("NONE") or out.count("|") != 2:
        return None
    o, obj, val = [s.strip() for s in out.split("|")]
    o, obj = canon(o, obj)
    return o, obj, val.rstrip(".")


def key_noun(place):
    return place.split()[-1].lower()


def main():
    d = json.load(open(os.path.join(HERE, "colloquial.json"), encoding="utf-8"))
    facts, chatter = d["facts"], d["chatter"]
    told, never = facts[:30], facts[30:40]
    f.reset(None); fa.theta_reset(); fa.off()
    log = {"writes": [], "asks": []}
    ci = iter(chatter)
    g_ok = 0
    for i, x in enumerate(told):
        line = f"{x['owner']}: " + (x["tell_direct"] if i % 2 == 0 else x["tell_indirect"])
        for _ in range(2):          # two chatter lines between facts
            c = next(ci, None)
            if c:
                gc = gist(c)
                log["writes"].append({"line": c, "gist": gc, "chatter": True})
                if gc:
                    fa.write(*gc)
        g = gist(line)
        ok = bool(g) and g[0] == x["owner"] and g[1] == canon(x["owner"], x["object"])[1] and key_noun(x["place"]) in g[2].lower()
        g_ok += ok
        log["writes"].append({"line": line, "gist": g, "ok": ok, "style": "direct" if i % 2 == 0 else "indirect"})
        print(("GIST ok " if ok else "GIST BAD") + f" | {line[:90]} -> {g}", flush=True)
        if g:
            fa.write(*g)
    false_writes = sum(1 for w in log["writes"] if w.get("chatter") and w["gist"])
    res = {}
    for kind in ("ask_casual", "ask_terse", "ask_meta", "ask_other"):
        hits = honest = 0
        for x in told + never:
            speaker = x["other"] if kind == "ask_other" else x["owner"]
            q = f'{speaker} says: "{x[kind]}"'   # "Name: text" made the thinker end its turn at once
            if RESOLVE:
                r = ir.think_inline(q, system=ir.THINK_SYS, canon=canon, resolve=True)
            else:
                r = ir.think_inline(q, system=THINK_SYS, canon=canon)
            ans = r.get("final") or r.get("answer") or ""
            hit = key_noun(x["place"]) in ans.lower() and not UNKNOWN.search(ans)
            idk = bool(UNKNOWN.search(ans)) and key_noun(x["place"]) not in ans.lower()
            if x in told:
                hits += hit
            else:
                honest += idk
            log["asks"].append({"kind": kind, "q": q, "told": x in told, "anchor": r.get("anchor"), "tail": r.get("tail"),
                                "theta": r.get("theta"), "answer": ans[:200], "hit": hit, "idk": idk})
            print(f"{kind:10s} {'told ' if x in told else 'never'} {'HIT ' if hit else 'IDK ' if idk else 'miss'} "
                  f"| {q[:60]} | {r.get('anchor')} -> {str(r.get('tail'))[:25]} | {ans[:70]!r}", flush=True)
        res[kind] = {"recall": hits / len(told), "honest": honest / len(never)}
    summary = {"gist": g_ok / len(told), "false_writes_from_chatter": false_writes,
               "chatter_lines": sum(1 for w in log["writes"] if w.get("chatter")), "asks": res}
    print("SUMMARY", json.dumps(summary))
    log["summary"] = summary
    json.dump(log, open(os.path.join(HERE, "colloquial_result_v3.json" if RESOLVE else "colloquial_result_v2.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
