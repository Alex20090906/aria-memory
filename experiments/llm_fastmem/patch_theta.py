"""Add familiarity theta on anchor units (H2) to fmem_anchor.py and a theta gate + answer phase to inline_recall.py."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

p = os.path.join(HERE, "fmem_anchor.py")
s = open(p, encoding="utf-8").read()
if "THETA =" not in s:
    s = s.replace('EMB = os.path.join(HERE, "token_embd_f16.npy")',
                  'EMB = os.path.join(HERE, "token_embd_f16.npy")\n'
                  'THETA = os.path.join(HERE, "theta.npy")      # familiarity engram on anchor units (H2)\n'
                  'THETA_TAU = float(os.environ.get("FMEM_THETA_TAU", "500"))   # decay per write event\n')
    s = s.replace('def write(owner, obj, value, dl=14.0, tau_w=1e9, ctx=None):\n'
                  '    n_mem, D, k, n, layers = f.read_header()\n',
                  'def theta_load(D):\n'
                  '    return np.load(THETA) if os.path.exists(THETA) else np.zeros(D, np.float32)\n\n\n'
                  'def theta_mark(anc, D):\n'
                  '    """one write event: every unit decays, the anchor units of this fact are set to 1"""\n'
                  '    th = theta_load(D) * np.float32(math.exp(-1 / THETA_TAU))\n'
                  '    th[anc] = 1.0\n'
                  '    np.save(THETA, th)\n\n\n'
                  'def familiarity(anc, D):\n'
                  '    return float(theta_load(D)[anc].mean())\n\n\n'
                  'def theta_reset():\n'
                  '    if os.path.exists(THETA):\n'
                  '        os.remove(THETA)\n\n\n'
                  'def write(owner, obj, value, dl=14.0, tau_w=1e9, ctx=None):\n'
                  '    n_mem, D, k, n, layers = f.read_header()\n')
    s = s.replace('    anc = anchor_units(E, P, st, toks, owner, obj, k)\n    pos = token_ids_of(toks, text)\n',
                  '    anc = anchor_units(E, P, st, toks, owner, obj, k)\n    theta_mark(anc, D)\n    pos = token_ids_of(toks, text)\n')
    open(p, "w", encoding="utf-8").write(s)
    print("fmem_anchor patched")

p = os.path.join(HERE, "inline_recall.py")
s = open(p, encoding="utf-8").read()
if "familiarity" not in s:
    s = s.replace('''    prompt = render(system, question, head)
    toks, resume, rows, _ = capture_raw(prompt)
    anc = fa.anchor_units(E, P, st, toks, owner, obj, k)
    fa.set_anchor(len(toks) - 1 + fa.ANCHOR_OFFSET, anc, k)
    tail = gen_raw(prompt, max_tokens)
    fa.off()
    return {"answer": head + " " + tail.strip(), "anchor": f"[{owner}|{obj}]", "head": head, "tail": tail.strip()}''',
'''    prompt = render(system, question, head)
    toks, resume, rows, _ = capture_raw(prompt)
    anc = fa.anchor_units(E, P, st, toks, owner, obj, k)
    fam = fa.familiarity(anc, D)
    if fam >= theta_gate:
        # phase 2: the memory completes the anchor (the place), still the same stream
        fa.set_anchor(len(toks) - 1 + fa.ANCHOR_OFFSET, anc, k)
        tail = gen_raw(prompt, 16, stop=["\\n"]).strip()
        fa.off()
    else:
        tail = "(nothing)"      # theta gate: never written -> no injection, the thinker sees an empty recall
    # phase 3: reasoning + answer, continuing the same text
    prompt3 = prompt + " " + tail + "\\nAria's answer to the person:"
    ans = gen_raw(prompt3, max_tokens, stop=["\\nPerson:"]).strip()
    return {"answer": head + " " + tail + "\\n" + ans, "anchor": f"[{owner}|{obj}]", "head": head, "tail": tail,
            "theta": round(fam, 3), "final": ans}''')
    s = s.replace("def think_inline(question, max_tokens=80, system=THINK_SYS):",
                  "def think_inline(question, max_tokens=80, system=THINK_SYS, theta_gate=0.5):")
    s = s.replace('    f.reset(None)\n    print("write:"', '    f.reset(None); fa.theta_reset()\n    print("write:"')
    s = s.replace('        print("   anchor:", r.get("anchor"), flush=True)',
                  '        print("   anchor:", r.get("anchor"), "theta:", r.get("theta"), flush=True)')
    open(p, "w", encoding="utf-8").write(s)
    print("inline_recall patched")
