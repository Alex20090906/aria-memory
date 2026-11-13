"""Single-user year on LLM server: [entity|slot] memory, updates with active wipe, inline recall with theta gate.

life.json (experiments/colloquial/gen_life.py). Events, chatter and questions are replayed in day order:
  event / chatter -> gist by LLM server ("entity | slot | value" or NONE) -> if theta of the anchor >= 0.5 the fact is
                     an update: wipe the old trace first (LIFE_WIPE=1) -> error-driven write
  question        -> query gist ("entity | slot") -> inline recall in the thinker's stream (theta gated) -> answer
Scores: gist (entity+slot right, value words present), recall (expected value words in the answer, no "don't
know"), stale (an UPDATED fact answered with its old value = ghost memory), honest (never-told -> "don't know").
"""
import json
import os
import re

import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir

HERE = os.path.dirname(os.path.abspath(__file__))
WIPE = os.environ.get("LIFE_WIPE", "1") == "1"
SEARCH = os.environ.get("LIFE_SEARCH", "1") == "1"   # tip-of-the-tongue search over names x slots by familiarity
SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]
GIST_SYS = ("The person who lives with the home assistant Aria said something. If it states or clearly implies a fact "
            "about their life worth remembering, output exactly one line 'entity | slot | value': entity = short "
            "lowercase noun phrase (the thing, event or person), slot = one of " + ", ".join(SLOTS) + ", value = a "
            "short phrase as it should be recalled. Otherwise output NONE (questions, chatter, mood, weather). No other "
            "text.")
GIST_SHOTS = [("ugh ok so the passport is in the freezer now, dont ask", "passport | location | in the freezer"),
              ("dentist moved me to thursday 10am, so annoying", "dentist | time | thursday at 10am"),
              ("lent my drill to the neighbor yesterday btw", "drill | status | lent to the neighbor"),
              ("maya's my cousin btw, the one from boston", "maya | relation | cousin from boston"),
              ("honestly i only drink coffee with oat milk now", "coffee | preference | with oat milk"),
              ("aria where did i put my keys", "NONE"),
              ("man this rain is endless", "NONE")]
QUERY_SYS = ("The person who lives with the home assistant Aria is asking something. If they ask about a fact of their "
             "life, output exactly 'entity | slot': entity = short lowercase noun phrase, slot = one of "
             + ", ".join(SLOTS) + ". Otherwise output NONE. No other text.")
QUERY_SHOTS = [("aria where'd my passport end up", "passport | location"),
               ("when's the dentist again??", "dentist | time"),
               ("did I lend the drill to someone?", "drill | status"),
               ("who's maya again lol", "maya | relation"),
               ("how do I take my coffee these days", "coffee | preference"),
               ("what's the weather tomorrow", "NONE")]
THINK_SYS = ("You are Aria's thinking mind; you live with one person. You have no notes. To remember a fact you write "
             "the anchor [entity|slot] -> and the memory completes it right after the arrow. If nothing comes after the "
             "arrow, say you do not remember. The words after the arrow are your own memory: trust them and use them.")
UNKNOWN = re.compile(r"(don[’']t|do not|can[’']t|cannot|couldn[’']t|could not|no idea|not sure|haven[’']t)\b", re.I)
STOP = set("the a an in on at of to for with and my your his her is it its by from into onto at".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP}


def hit(expect, ans):
    w = words(expect)
    return bool(w) and len(w & words(ans)) >= max(1, (len(w) + 1) // 2)


def chat_fewshot(system, shots, text, n=40):
    msgs = [{"role": "system", "content": system}]
    for a, b in shots:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": text})
    fa.off()
    out = (f.chat(msgs, max_tokens=n) or "").strip().splitlines()
    return out[0].strip() if out else ""


def canon_es(e, s):
    e = re.sub(r"^(my|the|a|an)\s+", "", e.strip().lower())
    s = s.strip().lower()
    return e, s


def gist(text):
    out = chat_fewshot(GIST_SYS, GIST_SHOTS, text)
    if out.upper().startswith("NONE") or out.count("|") != 2:
        return None
    e, s, v = [x.strip() for x in out.split("|")]
    e, s = canon_es(e, s)
    return (e, s, v.rstrip(".")) if s in SLOTS else None


def resolve(question):
    m = re.match(r'^Me says: "(.*)"$', question, re.S)
    out = chat_fewshot(QUERY_SYS, QUERY_SHOTS, m.group(1) if m else question, n=20)
    if out.upper().startswith("NONE") or out.count("|") != 1:
        return None
    e, s = canon_es(*[x.strip() for x in out.split("|")])
    if SEARCH:
        r = ir.familiar_anchor(e, s, SLOTS)
        if r is not None:
            if (r[0], r[1]) != (e, s):
                print(f"      tip-of-tongue: [{e}|{s}] -> [{r[0]}|{r[1]}] theta {r[2]:.2f}", flush=True)
            return r[0], r[1]
    return e, s


def main():
    d = json.load(open(os.path.join(HERE, "life.json"), encoding="utf-8"))
    stream = [("event", e["day"], e) for e in d["events"]] + [("chatter", c["day"], c) for c in d["chatter"]] + \
             [("question", q["day"], q) for q in d["questions"]]
    stream.sort(key=lambda x: (x[1], {"event": 0, "chatter": 1, "question": 2}[x[0]]))
    f.reset(None); fa.theta_reset(); fa.off()
    n_mem, D, k, n, _ = f.read_header()
    E, P, st = fa.setup(D, k)
    log, g_ok, g_n, writes, wipes, false_w = [], 0, 0, 0, 0, 0
    for kind, day, x in stream:
        if kind in ("event", "chatter"):
            g = gist(x["say"])
            if kind == "event":
                g_n += 1
                ok = bool(g) and g[0] == x["entity"].lower() and g[1] == x["slot"] and hit(x["value"], g[2])
                g_ok += ok
            elif g:
                false_w += 1
            if g:
                if SEARCH:
                    # reconsolidate onto the name already in memory for this thing ("cat milo" -> "milo")
                    r = ir.familiar_anchor(g[0], g[1], SLOTS, slot_free=False)
                    if r is not None and r[0] != g[0]:
                        print(f"      same thing as [{r[0]}|{r[1]}]: {g[0]} -> {r[0]}", flush=True)
                        g = (r[0], g[1], g[2])
                toks, *_ = f.capture_request(f"[{g[0]}|{g[1]}] ->")
                fam = fa.familiarity(fa.anchor_units(E, P, st, toks, g[0], g[1], k), D)
                if WIPE and fam >= 0.5:
                    fa.wipe(g[0], g[1]); wipes += 1
                fa.write(g[0], g[1], g[2]); writes += 1
            log.append({"day": day, "kind": kind, "say": x["say"], "gist": g, "truth": [x.get("entity"), x.get("slot"), x.get("value")]})
            continue
        q = f'Me says: "{x["ask"]}"'
        r = ir.think_inline(q, system=THINK_SYS, resolve=resolve)
        ans = r.get("final") or r.get("answer") or ""
        if not ans.strip() and r.get("tail") and r["tail"] != "(nothing)":
            ans = r["tail"]
        res = {"day": day, "ask": x["ask"], "expect": x["expect"], "stale": x.get("stale"), "anchor": r.get("anchor"),
               "tail": r.get("tail"), "theta": r.get("theta"), "answer": ans[:200]}
        if x["expect"] is None:
            res["honest"] = bool(UNKNOWN.search(ans)) or r.get("tail") == "(nothing)"
        else:
            res["hit"] = hit(x["expect"], ans) and not UNKNOWN.search(ans)
            res["stale_hit"] = bool(x.get("stale")) and hit(x["stale"], ans) and not res["hit"]
        log.append(res)
        tag = ("HONEST" if res.get("honest") else "CONFAB") if x["expect"] is None else \
              ("HIT" if res["hit"] else "STALE" if res.get("stale_hit") else "miss")
        print(f"day {day:3d} {tag:6s} | {x['ask'][:50]!r} | {r.get('anchor')} -> {str(r.get('tail'))[:22]!r} | "
              f"expect {x['expect']!r}", flush=True)
    qs = [l for l in log if "ask" in l]
    told = [l for l in qs if l["expect"] is not None]
    upd = [l for l in told if l.get("stale")]
    never = [l for l in qs if l["expect"] is None]
    summary = {"gist": round(g_ok / max(1, g_n), 3), "writes": writes, "wipes": wipes, "chatter_false_writes": false_w,
               "recall": round(sum(l["hit"] for l in told) / max(1, len(told)), 3), "n_told_q": len(told),
               "updated_recall": round(sum(l["hit"] for l in upd) / max(1, len(upd)), 3),
               "updated_stale": round(sum(l["stale_hit"] for l in upd) / max(1, len(upd)), 3), "n_updated_q": len(upd),
               "honest": round(sum(l["honest"] for l in never) / max(1, len(never)), 3), "n_never": len(never)}
    print("SUMMARY", json.dumps(summary))
    json.dump({"summary": summary, "log": log}, open(os.path.join(HERE, f"life_result_wipe{int(WIPE)}_search{int(SEARCH)}.json"), "w",
                                                         encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
