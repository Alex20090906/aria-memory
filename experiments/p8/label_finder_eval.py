"""P8 offline: how much recall is lost to label finding, and what a better label finder buys.

Replays the event log of a finished zh_v3 run (writes in order) against every scripted question, using the anchor
that LLM server itself parsed from the question during that run. No model is called: this measures retrieval only
(does the finder hand the thinker the right value?), and the false-memory risk on never-told / similar questions.

Finders:
  F0 exact      latest log entry whose label is exactly [entity, slot]   (current U15 lookup)
  F1 any_slot   same entity, any slot (prefer the asked slot, then latest)
  F2 fuzzy      entity matched by character overlap >= 0.5 in either direction, any slot
  F3 fuzzy+text F2, plus labels whose characters appear in the question text itself (catches a wrongly parsed anchor)

Answer policy: an exact label match is answered directly; any other match is an ask-back ("你是說 X 嗎？").

    python -I label_finder_eval.py <zh_result_*.json> [...]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FM = os.environ.get("P8_FM", os.path.join(HERE, "..", "llm_fastmem"))   # folder with timeline.json (P8_FM for another script set)
FILLER = set("的了在是我你他她它那個這個就啊啦喔欸嗎呢吧齁上面裡面下面一下有沒")
STOP_Q = set("Aria欸那個我的你知道嗎放哪裡啊？，、。有沒有跟說過現在什麼時候幾點")


def chars(s):
    return {c for c in re.sub(r"\W", "", s or "") if c not in FILLER}


def overlap(a, b):
    ca = chars(a)
    return len(ca & chars(b)) / max(1, len(ca))


def ent_score(a, b):
    return max(overlap(a, b), overlap(b, a))


def find(variant, e, s, qtext, log):
    """Return (value, label, exact) or None."""
    if variant == "F0":
        for ev in reversed(log):
            if ev["label"] == [e, s]:
                return ev["value"], ev["label"], True
        return None
    best, best_key = None, None
    for i, ev in enumerate(log):
        L, S = ev["label"]
        if variant == "F1":
            sc = 1.0 if L == e else 0.0
        else:
            sc = ent_score(e, L)
            if variant == "F3":
                qc = chars(qtext) - set(STOP_Q)
                lc = chars(L)
                if lc and len(lc & qc) / len(lc) >= 0.75:
                    sc = max(sc, len(lc & qc) / len(lc))
        if sc < (1.0 if variant == "F1" else 0.5):
            continue
        key = (round(sc, 3), S == s, i)                  # best match, then asked slot, then latest
        if best_key is None or key > best_key:
            best, best_key = ev, key
    if best is None:
        return None
    return best["value"], best["label"], best["label"] == [e, s]


def evaluate(path):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(FM, "timeline.json"), encoding="utf-8"))}
    log = [{"label": [w["fact"][0], w["fact"][1]], "value": w["fact"][2], "day": w["day"], "start": w["start"],
            "src": w["src"]} for w in d["writes"]]
    out = {}
    for variant in ("F0", "F1", "F2", "F3"):
        c = {"told": 0, "direct_ok": 0, "ask_ok": 0, "direct_wrong": 0, "ask_wrong": 0, "none": 0, "not_routed": 0,
             "never": 0, "never_direct": 0, "never_ask": 0}
        examples = []
        for q in d["questions"]:
            qq = q["q"]
            seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
            cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                           min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
            cand.sort(key=lambda a: -overlap(qq["entity"], (a["anchor"] or "").split("|")[0]))
            a = cand[0] if cand else None
            told = qq["expect"] is not None
            c["told" if told else "never"] += 1
            if a is None or not a["anchor"]:
                c["not_routed" if told else "never"] += 0
                if told:
                    c["not_routed"] += 1
                continue
            e, s = (a["anchor"].strip("[]").split("|") + [""])[:2]
            t0 = seg["start"]
            before = [ev for ev in log if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                                   and ev["start"] < t0)]
            r = find(variant, e, s, qq["text"], before)
            if told:
                if r is None:
                    c["none"] += 1
                    continue
                ok = overlap(qq["expect"], r[0]) >= 0.6
                c[("direct_" if r[2] else "ask_") + ("ok" if ok else "wrong")] += 1
                if not r[2]:
                    examples.append(("told", qq["entity"], f"[{e}|{s}]", "|".join(r[1]), r[0][:24], ok))
            else:
                if r is not None:
                    c["never_direct" if r[2] else "never_ask"] += 1
                    examples.append(("never", qq["entity"], f"[{e}|{s}]", "|".join(r[1]), r[0][:24], None))
        out[variant] = (c, examples)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        print("==", os.path.basename(p))
        res = evaluate(p)
        for v, (c, ex) in res.items():
            print(f"  {v}: told {c['told']} | direct ok {c['direct_ok']} ask-back ok {c['ask_ok']} | direct wrong "
                  f"{c['direct_wrong']} ask-back wrong {c['ask_wrong']} | none {c['none']} not routed {c['not_routed']}"
                  f" || never {c['never']}: direct {c['never_direct']} ask-back {c['never_ask']}")
        if "-v" in os.environ.get("P8_VERBOSE", ""):
            for row in res["F3"][1]:
                print("    ", row)
