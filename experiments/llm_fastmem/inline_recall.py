"""Inline recall (Gemini round 5): the thinker remembers inside its own generation stream.

The thinker is told that, to remember where someone's thing is, it writes an anchor "[owner|object] ->" and the
memory completes it.  Two phases, both through the raw-prompt path so the second one is a true continuation
(sending the partial assistant text as a chat message would close the turn and open a new one, so the last prompt
token would no longer be the arrow and nothing is injected):
  1. render the chat template by hand, generate until the thinker has written an anchor (stop at "->")
  2. append the head to the same raw prompt, capture its token ids, compute the anchor units from the
     owner/object tokens, set anchor.bin at the last prompt position, and continue: the engine injects the memory
     value at the anchor row and the shifted rows, so the thinker's own next words are the remembered place, and
     its reasoning continues from them.
Also the 2-hop test: a question whose answer needs the remembered fact plus one inference step.
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

import fmem_anchor as fa
import fmem_tool as f

THINK_SYS = ("You are Aria's thinking mind. You have no notes. To remember where a person's thing is, write the "
             "anchor [owner|object] -> on its own line and the memory completes it with the place; then go on "
             "reasoning from what you recalled and answer the person in one or two short sentences. If nothing "
             "comes after the arrow, say you do not remember. The words after the arrow are your own memory: trust them "
             "and use them.")
ANCHOR_RE = re.compile(r"\[([^|\]]{1,40})\|([^|\]]{1,60})\]\s*->\s*$")


def render(system, user, assistant_head=""):
    """Qwen3 chat template, thinking disabled (empty think block), assistant turn left open."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_head}")


def gen_raw(prompt, max_tokens, stop=None):
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": f.RAW + prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "llm_checkpoint": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if stop:
        body["stop"] = stop
    req = urllib.request.Request(f.URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def capture_raw(prompt):
    t0 = time.time()
    gen_raw(prompt, 1)
    for _ in range(100):
        if os.path.exists(f.CAP) and os.path.getmtime(f.CAP) >= t0 - 1:
            return f.read_capture()
        time.sleep(0.05)
    raise SystemExit("no capture file")


QUERY_SYS = ("Someone is talking to the home assistant Aria. If they ask where a person's thing is (or whether they "
             "told Aria where it is), output exactly 'owner | object': the owner is the person whose thing it is (the "
             "speaker when they say my/I; the named person when they ask about someone else, also when they refer to "
             "that person as he/she), the object is the bare noun. Otherwise output NONE. No other text.")
QUERY_SHOTS = [('Ken says: "ugh where did I leave my charger again??"', "Ken | charger"),
               ('Mia says: "Aria, my wallet?"', "Mia | wallet"),
               ('Tom says: "did I ever tell you where my ring is?"', "Tom | ring"),
               ('Sara says: "Omar is freaking out, any idea where he put his scarf?"', "Omar | scarf"),
               ('Dan says: "yo Aria, Lena lost her camera again, any clue?"', "Lena | camera"),
               ('Ivan says: "what is the weather tomorrow"', "NONE")]


def resolve_query(question):
    """The thinker's own first step: who and what is being asked about (same model, short chat call)."""
    msgs = [{"role": "system", "content": QUERY_SYS}]
    for a, b in QUERY_SHOTS:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": question})
    fa.off()
    out = (f.chat(msgs, max_tokens=20) or "").strip().splitlines()
    out = out[0] if out else ""
    if out.upper().startswith("NONE") or out.count("|") != 1:
        return None
    o, obj = [s.strip() for s in out.split("|")]
    return o, obj


ALIAS_SYS = ("List other short ways a person may have named this thing when they first mentioned it: shorter or longer "
             "forms, with or without a role word (sister, cat, my), synonyms. The input comes from speech recognition and "
             "may be misheard: also include the words it most likely was (same or similar sound, e.g. 被用藥實 -> 備用鑰匙). "
             "Same language as the input. One per line, at most 5, lowercase, nothing else.")


def familiar_anchor(entity, slot, slots, alias_lang_hint="", theta_gate=0.5, slot_free=True):
    """tip-of-the-tongue search: when [entity|slot] does not feel familiar, the thinker proposes other names for the
    thing and the familiarity theta over (name x slot) decides which cue to recall with. Returns (entity, slot, theta,
    tried) or None when nothing feels familiar."""
    n_mem, D, k, n, _ = f.read_header()
    E, P, st = fa.setup(D, k)
    cands = [(entity, slot)]
    first = fa.units_batch(E, P, st, cands, k)[0]
    th = fa.familiarity(first, D) if first is not None else 0.0
    if th >= theta_gate:
        return entity, slot, th, 1
    fa.off()
    msgs = [{"role": "system", "content": ALIAS_SYS}, {"role": "user", "content": entity + alias_lang_hint}]
    al = [a.strip(" -•*").lower() for a in (f.chat(msgs, max_tokens=60) or "").splitlines() if a.strip()][:5]
    names = [entity] + [a for a in al if a and a != entity and "|" not in a and "]" not in a]
    # switching slots freely turned never-told questions into confident answers about another property of the same
    # thing (familiarity-driven false memory). Only near-synonymous slots may stand in for each other.
    near = {"value": ["attribute"], "attribute": ["value"], "plan": ["time"], "time": ["plan"]}
    sl = [slot] + ([x for x in near.get(slot, []) if x in slots] if slot_free else [])
    pairs = [(nm, s) for nm in names for s in sl]
    units = fa.units_batch(E, P, st, pairs, k)
    best = None
    for (nm, s), u in zip(pairs, units):
        if u is None:
            continue
        t = fa.familiarity(u, D)
        # prefer the asked slot, then the asked name, on ties
        score = t + (0.05 if s == slot else 0) + (0.02 if nm == entity else 0)
        if t >= theta_gate and (best is None or score > best[3]):
            best = (nm, s, t, score)
    if best is None:
        return None
    return best[0], best[1], best[2], len(pairs) + 1


def clean_answer(ans, tail):
    lines = [l for l in ans.split("\n") if "->" not in l and "<|" not in l and l.strip()]
    ans = " ".join(lines).strip()
    if not ans and tail == "(nothing)":
        ans = "I don't remember."
    return ans


def think_inline(question, max_tokens=80, system=THINK_SYS, theta_gate=0.5, canon=None, resolve=False, lookup=None):
    """canon: optional (owner, obj) -> (owner, obj) normaliser; the key is then computed from the canonical names
    (same "[owner|obj] ->" frame as at write time), whatever spelling the thinker used in its own stream.
    resolve: the anchor is decided by resolve_query() and placed at the head of the thinker's answer, instead of
    letting the thinker write it (writing it freely picked the asker or copied an example name for third-person asks)."""
    n_mem, D, k, n, layers = f.read_header()
    E, P, st = fa.setup(D, k)
    fa.off()
    if resolve:
        r = resolve(question) if callable(resolve) else resolve_query(question)
        if r is None:
            return {"answer": "", "anchor": None, "tail": None, "theta": None, "final": ""}
        if canon is not None:
            r = canon(*r)
        head = f"[{r[0]}|{r[1]}] ->"
    else:
        head = gen_raw(render(system, question), max_tokens, stop=["->"]).rstrip()
    if not head.endswith("->"):
        head += " ->"
    m = ANCHOR_RE.search(head)
    if not m:
        return {"answer": head, "anchor": None}
    owner, obj = m.group(1).strip(), m.group(2).strip()
    prompt = render(system, question, head)
    toks, resume, rows, _ = capture_raw(prompt)
    if canon is not None and canon(owner, obj) != (owner, obj):
        owner, obj = canon(owner, obj)
        ktoks, *_ = f.capture_request(f"[{owner}|{obj}] ->")
        anc = fa.anchor_units(E, P, st, ktoks, owner, obj, k)
    else:
        anc = fa.anchor_units(E, P, st, toks, owner, obj, k)
    fam = fa.familiarity(anc, D)
    exact = lookup(owner, obj) if (lookup is not None and fam >= theta_gate) else None
    if exact:
        # U15: the network produced the label and says "I know this" (theta); the detail is read from the
        # append-only event log at exactly that label (the log cannot be searched, only opened at a label)
        tail = exact
    elif fam >= theta_gate:
        # phase 2: the memory completes the anchor (the place), still the same stream
        fa.set_anchor(len(toks) - 1 + fa.ANCHOR_OFFSET, anc, k, anchor_tok=int(toks[-1]), weights=fa.read_weights(anc, D, k))
        tail = gen_raw(prompt, 16, stop=["\n"]).strip()
        fa.off()
    else:
        tail = "(nothing)"      # theta gate: never written -> no injection, the thinker sees an empty recall
    # phase 3: reasoning + answer, continuing the same text
    prompt3 = prompt + " " + tail + "\nAria's answer to the person:"
    ans = gen_raw(prompt3, max_tokens, stop=["\nPerson:", "<|im_end|>", "\n\n"]).strip()
    ans = clean_answer(ans.split("<|im_end|>")[0].strip(), tail)
    if ans == "I don't remember." and re.search(r"[一-鿿]", question):
        ans = "我不記得了。"
    return {"answer": head + " " + tail + "\n" + ans, "anchor": f"[{owner}|{obj}]", "head": head, "tail": tail,
            "theta": round(fam, 3), "final": ans}


if __name__ == "__main__":
    f.reset(None); fa.theta_reset()
    print("write:", fa.write("Leo", "passport", "in the freezer"))
    print("write:", fa.write("Mia", "watch", "in the garage drawer"))
    qs = ["Where is Leo's passport?",
          "Is Leo's passport likely to get frosted? Think it through.",
          "Leo wants to fly tomorrow morning. Will he find his passport quickly, and should he worry about its condition?",
          "Mia's watch stopped. Where should she look for it, and is that place likely to be dusty?",
          "Where is Omar's wallet?"]
    for q in qs * 2:
        r = think_inline(q)
        print("Q:", q)
        print("A:", r["answer"])
        print("   anchor:", r.get("anchor"), "theta:", r.get("theta"), flush=True)
