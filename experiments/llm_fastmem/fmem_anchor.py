"""Anchor keys for the LLM server fast-weight memory (runs on the AI PC with LLM server's Python, numpy only).

Key = layer-0 bilinear binding of owner and object (validated locally: C0 97.5%, natural question C1 92.5%):
    x_s = adapted mean embedding of the owner tokens,  x_o = of the object tokens
    s   = (x_s @ P) * roll(x_o @ P, 1);   anchor units = top-k(s)
Embeddings come straight from the GGUF token table (IQ4_XS, dequantized here); no hidden state is involved, so
the key is the same whatever conversation precedes the anchor.
Answer rows (m = 1, 2, ... after "->") use (anchor + (m*1000003 + token*7919) mod D) mod D, as the engine does.

  write   [owner|object] -> value.       : capture token ids, targets from the API's top-20, delta rule into B
  recall  <any text ending with the anchor> : write anchor.bin, then let the engine complete it
"""
import json
import math
import os
import struct
import sys

import numpy as np

import fmem_tool as f

HERE = os.path.dirname(os.path.abspath(__file__))
ANC = os.path.join(f.DATA, "anchor.bin")
PCACHE = os.path.join(f.DATA, "anchor_P.npy")
STATS = os.path.join(f.DATA, "anchor_stats.npz")
EMB = os.path.join(f.DATA, "token_embd_f16.npy")
THETA = os.path.join(f.DATA, "theta.npy")      # familiarity engram on anchor units (H2)
THETA_TAU = float(os.environ.get("FMEM_THETA_TAU", "500"))   # decay per write event
THETA_TAU_SLOW = float(os.environ.get("FMEM_THETA_TAU_SLOW", "5000"))
TAU_FILE = os.path.join(f.DATA, "tau.npy")       # per-unit weight time constants written by sleep.py

KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


def dequant_iq4_xs(raw, n_blocks):
    """ggml IQ4_XS: per 256 values d(fp16) scales_h(u16) scales_l[4] qs[128] = 136 bytes"""
    b = np.frombuffer(raw, np.uint8).reshape(n_blocks, 136)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    sh = b[:, 2:4].copy().view(np.uint16).astype(np.int32)[:, 0]
    sl = b[:, 4:8].astype(np.int32)
    qs = b[:, 8:136].astype(np.int32)
    y = np.empty((n_blocks, 256), np.float32)
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        q = qs[:, ib * 16:ib * 16 + 16]
        y[:, ib * 32:ib * 32 + 16] = dl[:, None] * KV[q & 0xF]
        y[:, ib * 32 + 16:ib * 32 + 32] = dl[:, None] * KV[q >> 4]
    return y.reshape(-1)


def build_embed():
    path, off, t = f.find_tensor("token_embd.weight")
    n, vocab = t.shape
    assert t.type_name == "IQ4_XS", t.type_name
    row_bytes = n // 256 * 136
    E = np.lib.format.open_memmap(EMB, mode="w+", dtype=np.float16, shape=(vocab, n))
    with open(path, "rb") as fh:
        fh.seek(off)
        for r0 in range(0, vocab, 8192):
            nr = min(8192, vocab - r0)
            E[r0:r0 + nr] = dequant_iq4_xs(fh.read(nr * row_bytes), nr * n // 256).reshape(nr, n).astype(np.float16)
    E.flush()
    print("token embeddings", vocab, "x", n)


def setup(D, k, seed=1):
    if not os.path.exists(EMB):
        build_embed()
    E = np.load(EMB, mmap_mode="r")
    n = E.shape[1]
    pc = PCACHE if D == 16384 else PCACHE.replace(".npy", f"_{D}.npy")     # one projection per memory size
    if not os.path.exists(pc):
        rng = np.random.default_rng(seed)
        np.save(pc, (rng.standard_normal((n, D), dtype=np.float32) / math.sqrt(n)).astype(np.float16))
    if not os.path.exists(STATS):
        rng = np.random.default_rng(seed + 1)
        rows = E[np.sort(rng.integers(0, E.shape[0], 2000))].astype(np.float32)
        rows = (rows - rows.mean(1, keepdims=True)) / (rows.std(1, keepdims=True) + 1e-5)
        np.savez(STATS, mu=rows.mean(0), sd=rows.std(0) + 1e-3)
    return E, np.load(pc, mmap_mode="r"), np.load(STATS)


def token_ids_of(toks, text):
    """indices of the prompt tokens inside the last occurrence of `text`"""
    return f.fact_positions(toks, text)


KEYFIX = os.environ.get("FMEM_KEYFIX", "1") == "1"


def key_token_ids(toks, owner, obj):
    """Token ids of owner and object read inside the LAST "[owner|obj]" key, not at the last place each word occurs.
    E39 found that a value repeating a key word ("[Noah|umbrella] -> in the umbrella stand.", "pen" in "pencil cup")
    moved the write key onto the value, so those facts were stored under a key that recall never builds. Where the
    key words do not occur in the value the token spans are the same as before."""
    vb = f.id_to_bytes()
    pieces = [vb.get(int(t), b"") for t in toks]
    starts, pos = [], 0
    for p_ in pieces:
        starts.append(pos); pos += len(p_)
    full = b"".join(pieces)
    i = full.rfind(f"[{owner}|{obj}]".encode())
    if i < 0:
        return None
    o_a = i + 1; o_b = o_a + len(owner.encode())
    s_a = o_b + 1; s_b = s_a + len(obj.encode())
    ids_s = [int(toks[t]) for t, s in enumerate(starts) if o_a <= s < o_b]
    ids_o = [int(toks[t]) for t, s in enumerate(starts) if s_a <= s < s_b]
    return (ids_s, ids_o) if ids_s and ids_o else None


def anchor_units(E, P, st, toks, owner, obj, k):
    kt = key_token_ids(toks, owner, obj) if KEYFIX else None
    if kt:
        ids_s, ids_o = kt
    else:
        ids_s = [int(toks[i]) for i in token_ids_of(toks, owner)]
        ids_o = [int(toks[i]) for i in token_ids_of(toks, obj)]
    return anchor_units_ids(E, P, st, ids_s, ids_o, k, owner, obj)


# ---------------------------------------------------------------- dual-bank key (Gemini round 8)
# Bank 1 (units 0..D/2-1): meaning of the entity (token embeddings), k/2 units.
# Bank 2 (units D/2..D-1): sound of the entity (toneless pinyin syllables with easily confused sounds merged), k/2.
# The slot does not multiply a hashed entity any more: each slot has its own fixed random permutation and signs of
# the bank coordinates, so similar entities stay similar within a slot and slots stay apart.
_feat_cache, _perm_cache = {}, {}
CONFUSE = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]


def _rand_vec(key, n):
    v = _feat_cache.get((key, n))
    if v is None:
        import hashlib
        seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
        _feat_cache[(key, n)] = v
    return v


def _perm(slot, bank, n):
    p = _perm_cache.get((slot, bank, n))
    if p is None:
        import hashlib
        seed = int(hashlib.md5(f"{slot}|{bank}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        p = (rng.permutation(n), rng.choice(np.array([-1.0, 1.0], np.float32), n))
        _perm_cache[(slot, bank, n)] = p
    return p


def phonetic_units(entity):
    from pypinyin import lazy_pinyin, Style
    syl = []
    for s in lazy_pinyin(entity.lower(), style=Style.NORMAL, errors=lambda x: x.split()):
        s = s.strip()
        if not s:
            continue
        for a, b in CONFUSE:
            s = s.replace(a, b)
        syl.append(s)
    return syl


def phonetic_vec(entity, n):
    syl = phonetic_units(entity)
    feats = [f"u:{s}" for s in syl] + [f"b:{a}_{b}" for a, b in zip(syl, syl[1:])]
    if not feats:
        return np.zeros(n, np.float32)
    v = sum(_rand_vec(x, n) for x in feats)
    return v / (np.linalg.norm(v) + 1e-6) * math.sqrt(n)


def dual_units(xs, entity, slot, P32, k):
    D = P32.shape[1]
    h = D // 2
    kb = k // 2
    sem = xs @ P32[:, :h]
    sem = sem / (sem.std() + 1e-6)
    ph = phonetic_vec(entity, h)
    out = []
    for bank, z in ((0, sem), (1, ph)):
        perm, sign = _perm(slot, bank, h)
        s = sign * z[perm]
        out.append(np.argpartition(-s, kb)[:kb] + bank * h)
    return np.sort(np.concatenate(out)).astype(np.int32)


def units_batch(E, P, st, pairs, k):
    """anchor units for many (owner, obj) pairs from ONE tokenisation request: the lines "[o|s] ->" are tokenised
    together; each line is preceded by a newline exactly as a single "Memory notes.\\n[o|s] ->" capture is, so the
    token boundaries (and keys) are the same as the one-at-a-time path."""
    text = "\n".join(f"[{o}|{s}] ->" for o, s in pairs)
    toks, *_ = f.capture_request(text)
    vb = f.id_to_bytes()
    pieces = [vb.get(int(t), b"") for t in toks]
    starts, pos = [], 0
    for p_ in pieces:
        starts.append(pos); pos += len(p_)
    full = b"".join(pieces)
    base = full.rfind(text.encode())
    out, cur = [], base
    for o, s in pairs:
        line = f"[{o}|{s}] ->".encode()
        i = full.find(line, cur)
        cur = i + len(line)
        o_a = i + 1; o_b = o_a + len(o.encode())
        s_a = o_b + 1; s_b = s_a + len(s.encode())
        ids_s = [int(toks[t]) for t, st_ in enumerate(starts) if o_a <= st_ < o_b]
        ids_o = [int(toks[t]) for t, st_ in enumerate(starts) if s_a <= st_ < s_b]
        out.append(anchor_units_ids(E, P, st, ids_s, ids_o, k, o, s) if ids_s and ids_o else None)
    return out


def anchor_units_ids(E, P, st, ids_s, ids_o, k, ent=None, slot=None):
    def emb(ids):
        x = E[ids].astype(np.float32)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)
        return (x.mean(0) - st["mu"]) / st["sd"]
    xs, xo = emb(ids_s), emb(ids_o)
    P32 = np.asarray(P, np.float32)
    D = P32.shape[1]
    if KEY == "dual":
        return dual_units(xs, ent, slot, P32, k)
    if KEY == "cross":
        # Gemini round 6, A1: cross-shifted clipped bilinear key. Each side's projection is rolled by a hash of the
        # OTHER entity's tokens and squashed by tanh, so the same owner with another object lands on new units
        # (no shared "hub" dimensions between facts that share an entity).
        hs = sum(t * 7919 for t in ids_s) % D
        ho = sum(t * 1000003 for t in ids_o) % D
        a = np.tanh((xs @ P32) / CROSS_SIGMA)
        b = np.tanh((xo @ P32) / CROSS_SIGMA)
        s = np.roll(a, ho) * np.roll(b, hs)
    else:
        s = (xs @ P32) * np.roll(xo @ P32, 1)
    return np.sort(np.argpartition(-s, k)[:k]).astype(np.int32)


def shifted(anc, m, tok, D):
    # must match the engine (LLM_FMEM_SHIFT_NOTOK): with NOTOK the key depends on the step m only
    return (anc + ((m * 1000003 + (0 if SHIFT_NOTOK else int(tok) * 7919)) % D)) % D


def set_anchor_rel(back, units, k):
    """the anchor is `back` rows before the first row of the next request's window (its last prompt token)"""
    set_anchor(-(3 + int(back)), units, k, anchor_tok=-1)


def gap_nats(dist, y):
    """prior gap of target y: best competitor logit minus y's logit (from top-20 probabilities)"""
    if not dist:
        return 0.0
    d = dict(dist)
    py = d.get(y, min(d.values()) * 0.5)
    comp = max((p for t, p in dist if t != y), default=py)
    return math.log(comp + 1e-12) - math.log(py + 1e-12)


_last_mtime = [0]
# the engine's row position of the prompt's last token, relative to len(prompt tokens) - 1 (measured; see spell.py)
ANCHOR_OFFSET = int(os.environ.get("FMEM_ANCHOR_OFFSET", "0"))
ERR_DRIVEN = os.environ.get("FMEM_ERR_DRIVEN", "1") == "1"
WRITE_FRAME = os.environ.get("FMEM_WRITE_FRAME", "pre")   # pre | think (targets computed in the thinker's frame)
KEY = os.environ.get("FMEM_KEY", "bilinear")          # bilinear | cross (A1)
CROSS_SIGMA = float(os.environ.get("FMEM_CROSS_SIGMA", "1.0"))
CONF_PMIN = float(os.environ.get("FMEM_CONF_PMIN", "0"))   # >0: weakest answer token must reach this (H14b)
CONF_TH = float(os.environ.get("FMEM_CONF_TH", "0"))   # >0: say "not remembered" when the recalled first token is weak
READ_WEIGHTS = os.environ.get("FMEM_READ_WEIGHTS", "1") == "1"   # theta-weighted read (dual keys, engine STRFANC3)
SHIFT_NOTOK = os.environ.get("FMEM_SHIFT_NOTOK", "0") == "1"   # answer keys = anchor + hash(m) (engine flag too)
ERR_ALL = os.environ.get("FMEM_ERR_ALL", "0") == "1"           # error-driven read at every answer step (relative anchor)
MARGIN = os.environ.get("FMEM_MARGIN", "0") == "1"             # write strength = clip(prior gap + 8, 8, 24), gap from
                                                               # the "Memory notes." frame and the thinker's chat frame


def set_anchor(anchor_pos, units, k, anchor_tok=None, weights=None):
    """anchor_tok given: STRFANC2, the engine locks the anchor at the first window row carrying that token
    (the arrow, last prompt token), so no absolute position is needed. anchor_pos <= -3 means "relative to the
    first window row" (set_anchor_rel). weights given: STRFANC3, per-unit read weights."""
    import time
    pos = int(anchor_pos) if int(anchor_pos) <= -3 else -2
    with open(ANC, "wb") as fh:
        if anchor_tok is None and weights is None:
            fh.write(b"STRFANC1" + struct.pack("<2i", int(anchor_pos), k) + np.asarray(units, np.int32).tobytes())
        elif weights is None:
            fh.write(b"STRFANC2" + struct.pack("<3i", pos, k, int(anchor_tok)) + np.asarray(units, np.int32).tobytes())
        else:
            fh.write(b"STRFANC3" + struct.pack("<3i", pos, k, int(anchor_tok if anchor_tok is not None else -1))
                     + np.asarray(units, np.int32).tobytes() + np.asarray(weights, np.float32).tobytes())
    # the engine notices a change by whole-second mtime: make it strictly increase even within one second
    t = max(time.time(), _last_mtime[0] + 1.0)
    _last_mtime[0] = t
    os.utime(ANC, (t, t))


def theta_load(D):
    return np.load(THETA) if os.path.exists(THETA) else np.zeros(D, np.float32)


def theta_mark(anc, D):
    """one write event: every unit decays, the anchor units of this fact are set to 1"""
    th = theta_load(D) * np.float32(math.exp(-1 / THETA_TAU))
    if os.path.exists(TAU_FILE):
        # units promoted by sleep (slow tau) keep their familiarity for much longer too
        slow = np.load(TAU_FILE) >= 500
        th0 = theta_load(D)
        th[slow] = th0[slow] * np.float32(math.exp(-1 / THETA_TAU_SLOW))
    th[anc] = 1.0
    np.save(THETA, th)


def read_weights(anc, D, k):
    """theta-weighted read (Gemini round 8): unmatched units (theta ~ 0) are silenced and the matched ones scaled up
    so the injected value keeps its full strength when only one bank of the dual key matches."""
    if KEY != "dual" or not READ_WEIGHTS:
        return None
    th = np.clip(theta_load(D)[np.asarray(anc)], 0.0, 1.0).astype(np.float32)
    keff = float(th.sum())
    return (th * k / max(keff, 4.0)).astype(np.float32)


def familiarity(anc, D):
    if ECHO:
        return echo_familiarity(anc, D)
    th = theta_load(D)[anc]
    if KEY == "dual":
        # max over banks: a misheard word can still sound familiar, a synonym can still mean something familiar
        a = np.asarray(anc)
        m = [float(th[a < D // 2].mean()) if (a < D // 2).any() else 0.0,
             float(th[a >= D // 2].mean()) if (a >= D // 2).any() else 0.0]
        return max(m)
    return float(th.mean())


# U16 question 2, Kaggle round 3 (familiarity AUC 1.00 where theta fell to 0.74): every label also stores its own
# random +-1 signature on its anchor units ("key echo"); familiarity = cosine between the summed echo rows (with the
# same per-row scales as the content, so the echo fades and is downscaled together with what it stands for) and the
# query's signature. A never-heard label only meets other labels' signatures: noise around 0.
ECHO = os.environ.get("FMEM_ECHO", "0") == "1"
ECHO_TH = float(os.environ.get("FMEM_ECHO_TH", "0.5"))     # cosine that counts as "heard"; maps to the 0.5 gate
ECHO_L = 128
ECHO_FILE = os.path.join(f.DATA, "echo.npy")


_G = {}


def echo_sig(anc, D):
    """the label's signature is a function of its anchor units (sign of a fixed random projection of the unit set),
    so a misheard or alias form that lands on the same units has the same signature, and one that shares only a few
    units has a nearly unrelated one"""
    if D not in _G:
        _G[D] = np.random.default_rng(777).standard_normal((D, ECHO_L)).astype(np.float32)
    return np.sign(_G[D][np.asarray(anc)].sum(0) + 1e-9).astype(np.float32)


def echo_load(D):
    return np.load(ECHO_FILE) if os.path.exists(ECHO_FILE) else np.zeros((D, ECHO_L), np.float32)


def echo_familiarity(anc, D):
    EB = echo_load(D)
    rs = f.rs_load(D)
    a = np.asarray(anc)

    def cos(units):
        z = (EB[units] * rs[units][:, None]).sum(0)
        return float(z @ echo_sig(units, D) / (np.linalg.norm(z) * math.sqrt(ECHO_L) + 1e-9))
    if KEY == "dual":
        # each bank has its own signature (meaning units, sound units): a misheard word matches through the sound bank
        c = max(cos(a[a < D // 2]) if (a < D // 2).any() else 0.0, cos(a[a >= D // 2]) if (a >= D // 2).any() else 0.0)
    else:
        c = cos(a)
    return 0.5 * c / ECHO_TH


def theta_reset():
    if os.path.exists(ECHO_FILE):
        os.remove(ECHO_FILE)
    for q in (THETA, TAU_FILE):
        if os.path.exists(q):
            os.remove(q)


def write(owner, obj, value, dl=14.0, tau_w=1e9, ctx=None):
    n_mem, D, k, n, layers = f.read_header()
    lay, _ = f.mem_layout(n_mem, D, n)
    E, P, st = setup(D, k)
    U = np.load(f.UCACHE, mmap_mode="r")
    Ubar = np.load(f.UMEAN)
    text = f"[{owner}|{obj}] -> {value}."
    toks, resume, rows, _ = f.capture_request(text)
    anc = anchor_units(E, P, st, toks, owner, obj, k)
    theta_mark(anc, D)
    pos = token_ids_of(toks, text)
    vb = f.id_to_bytes()
    a_end = max(i for i in pos if b"->" in vb.get(int(toks[i]), b""))
    mm = np.memmap(f.MEM, dtype=np.uint8, mode="r+")
    B = np.ndarray((D, n), np.float16, mm, lay[0][3])
    rs = f.rs_load(D)          # per-row scales (all ones unless FMEM_ROWSCALE_FILE)
    rs_old = rs.copy()
    touched = set()
    n_written = 0
    if ERR_DRIVEN:
        # true delta rule: the target distribution is read WITH the current memory injected at the anchor, so the
        # write cancels interference from earlier facts on shared units instead of stacking on top of it
        set_anchor(a_end + ANCHOR_OFFSET, anc, k, anchor_tok=int(toks[a_end]))
    head = f.PRE
    if WRITE_FRAME == "think":
        # encoding specificity: compute the targets in the frame where recall happens (the thinker's own answer
        # turn after a generic question), so the value outweighs the thinker's prior there, not only in "Memory notes."
        import inline_recall as ir
        head = ir.render(ir.THINK_SYS, f'{owner} says: "where is my {obj}?"')
    chat_head = None
    if MARGIN:
        import inline_recall as ir
        chat_head = ir.render(ir.THINK_SYS, f'{owner} says: "where is my {obj}?"')
    for t in range(a_end, pos[-1]):
        y = int(toks[t + 1])
        body = b"".join(vb.get(int(x), b"") for x in toks[pos[0]:t + 1]).decode("utf-8", "replace")
        prefix = head + body
        if ERR_DRIVEN and ERR_ALL:
            set_anchor_rel(t - a_end, anc, k)      # memory on at this step too (the engine counts m from the anchor)
        dist = f.next_dist(prefix)
        ps = np.array([p for _, p in dist], np.float32)
        if len(dist) == 0:      # model wanted to end the turn here: no visible competitor, push from the mean
            Eu = Ubar.astype(np.float32)
        else:
            Eu = (ps[:, None] * np.stack([U[i].astype(np.float32) for i, _ in dist])).sum(0) + (1 - ps.sum()) * Ubar
        p_y = dict(dist).get(y, 0.0)
        du = U[y].astype(np.float32) - Eu
        dl_t = dl
        if MARGIN:
            dist_c = f.next_dist(chat_head + body)
            g = max(gap_nats(dist, y), gap_nats(dist_c, y) if dist_c else -99.0)
            dl_t = min(24.0, max(8.0, g + 8.0))
            p_y = min(p_y, dict(dist_c).get(y, 0.0) if dist_c else p_y)
        tgt = dl_t * (1 - p_y) * du / (float(du @ du) + 1e-6)
        u = anc if t == a_end else shifted(anc, t - a_end, toks[t], D)
        # materialize the row scale first (effective row = scale * stored row), then add
        B[u] = (B[u].astype(np.float32) * rs[u][:, None] + tgt / math.sqrt(k)).astype(np.float16)
        rs[u] = 1.0
        touched.update(int(x) for x in np.asarray(u).ravel())
        n_written += 1
    if tau_w < 1e8:
        if f.ROWSCALE:
            rs *= np.float32(math.exp(-1 / tau_w))
        else:
            B *= np.float16(math.exp(-1 / tau_w))
            touched = None
    mm.flush(); del B, mm
    if ECHO:
        assert f.ROWSCALE, "echo familiarity needs the per-row scales (FMEM_ROWSCALE_FILE)"
        EB = echo_load(D)
        ch = np.array(sorted(touched), np.int64)
        EB[ch] *= rs_old[ch][:, None]        # these rows' scales were reset to 1: keep their echo where it was
        a = np.asarray(anc)
        if KEY == "dual":
            for part in (a[a < D // 2], a[a >= D // 2]):
                if len(part):
                    EB[part] += echo_sig(part, D) / math.sqrt(k)
        else:
            EB[a] += echo_sig(a, D) / math.sqrt(k)
        np.save(ECHO_FILE, EB)
    f.rs_save(rs)
    f.touch_mem(touched)
    if ERR_DRIVEN:
        set_anchor(-1, np.zeros(k, np.int32), k)
    return n_written


def wipe(owner, obj, M=8, keep=0.05):
    """active reconsolidation (Gemini round 7): before an UPDATE of [owner|obj], shrink the old answer trace on all
    answer steps m = 0..M-1 (with no-previous-token keys the trace is known without knowing the old value)"""
    assert SHIFT_NOTOK, "wipe needs answer keys without the previous token"
    n_mem, D, k, n, layers = f.read_header()
    lay, _ = f.mem_layout(n_mem, D, n)
    E, P, st = setup(D, k)
    toks, *_ = f.capture_request(f"[{owner}|{obj}] ->")
    anc = anchor_units(E, P, st, toks, owner, obj, k)
    units = np.unique(np.concatenate([anc] + [shifted(anc, m, 0, D) for m in range(1, M)]))
    if f.ROWSCALE:
        rs = f.rs_load(D)
        rs[units] *= np.float32(keep)
        f.rs_save(rs)
        return len(units)
    mm = np.memmap(f.MEM, dtype=np.uint8, mode="r+")
    B = np.ndarray((D, n), np.float16, mm, lay[0][3])
    B[units] = (B[units].astype(np.float32) * keep).astype(np.float16)
    mm.flush(); del B, mm
    f.touch_mem(units)
    return len(units)


def recall(prompt, owner, obj, max_tokens=8):
    """prompt must end with '[owner|obj] ->' (raw text, no template)"""
    n_mem, D, k, n, layers = f.read_header()
    E, P, st = setup(D, k)
    set_anchor(-1, np.zeros(k, np.int32), k)                       # first pass: no injection, just the token ids
    toks, resume, rows, _ = f.capture_request(prompt)
    anc = anchor_units(E, P, st, toks, owner, obj, k)
    set_anchor(len(toks) - 1 + ANCHOR_OFFSET, anc, k, anchor_tok=int(toks[-1]), weights=read_weights(anc, D, k))
    if CONF_TH > 0:
        # content confidence (H14): familiarity says "I know this", but if the memory does not make the first answer
        # token clearly likely, the trace is corrupted -> treat it as not remembered instead of answering confidently
        dist = f.next_dist(prompt)
        top = max((p for _, p in dist), default=0.0)
        if top < CONF_TH:
            set_anchor(-1, np.zeros(k, np.int32), k)
            return ""
    if CONF_PMIN > 0:
        # H14b (Kaggle rounds 1-2: the weakest answer token separates true from corrupted memories best, AUC 0.75-0.92;
        # the first token alone 0.62-0.68): answer only if every token of the recalled answer is confidently produced
        text, probs = chat_lp(prompt, max_tokens)
        global LAST_PMIN
        LAST_PMIN = min(probs) if probs else 0.0
        if LAST_PMIN < CONF_PMIN:
            set_anchor(-1, np.zeros(k, np.int32), k)
            return ""
        return text
    return f.chat([{"role": "user", "content": f.RAW + prompt}], max_tokens=max_tokens)


LAST_PMIN = None


def chat_lp(prompt, max_tokens=8):
    """greedy completion with the probability of every produced token, up to the end of the answer phrase"""
    import json as _json
    import urllib.request as _ur
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": f.RAW + prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "logprobs": True, "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "llm_checkpoint": False}
    req = _ur.Request(f.URL, data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with _ur.urlopen(req, timeout=600) as r:
        out = _json.load(r)
    ch = out["choices"][0]
    text = (ch.get("message") or {}).get("content") or ""
    probs = []
    for e in ((ch.get("logprobs") or {}).get("content") or []):
        tok = e.get("token", "")
        if probs and tok.strip() == "":
            break                       # a bare newline after the phrase: the answer ended
        probs.append(math.exp(e["logprob"]))
        if "." in tok or "\n" in tok:
            break                       # the answer phrase ended
    return text, probs


def off():
    n_mem, D, k, n, layers = f.read_header()
    set_anchor(-1, np.zeros(k, np.int32), k)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "setup":
        n_mem, D, k, n, layers = f.read_header()
        setup(D, k)
    elif cmd == "write":
        print(write(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "recall":
        print(recall(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "off":
        off()
