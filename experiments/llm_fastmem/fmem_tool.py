"""Memory compiler for the LLM server fast-weight engine (runs on the AI PC with LLM server's own Python: numpy only).

  init       build memory.bin (fixed random P, B = 0, mu = 0, sd = 1) and the unembedding cache
  calibrate  run ordinary sentences with capture on and set mu, sd (sensory adaptation)
  write      experience one fact: read which units its tokens used, add the output-direction target
  ask        a chat question (to test recall)

The engine must run with LLM_FMEM_FILE=<memory.bin> and LLM_FMEM_CAPTURE=<capture.bin>.
Write rule (validated on Qwen2.5-1.5B, experiments/memory/compile_ctx.py --target logit --no_prob 1):
for each fact position t with next token y:  B_l[u] += eta / sqrt(k) * gain * unit(U[y] - mean U) / n_mem
for the k units u the token used at memory layer l; afterwards every B decays by exp(-1 / tau_w).
"""
import argparse
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# every path can be overridden from the environment (cloud runs on Linux); the defaults are the AI PC's
LLM = os.environ.get("FMEM_LLM", r"<thinker-workdir>\LLMHome\Local\LLM server\terminal\0.1.53")
GGUF = os.environ.get("FMEM_GGUF", r"<thinker-workdir>\Models\llm\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
GGUF2 = GGUF.replace("00001-of-00002", "00002-of-00002")
TOKENIZER = os.environ.get("FMEM_TOKENIZER", r"<thinker-workdir>\Models\llm\packs\iq3_s\tokenizer\vocab.json")
DATA = os.environ.get("FMEM_DATA", HERE)          # memory.bin, capture.bin, caches
MEM = os.path.join(DATA, "memory.bin")
CAP = os.path.join(DATA, "capture.bin")
UCACHE = os.path.join(DATA, "unembed_f16.npy")
UMEAN = os.path.join(DATA, "unembed_mean.npy")
URL = os.environ.get("FMEM_URL", "http://127.0.0.1:8093/v1/chat/completions")


# ---------------------------------------------------------------- GGUF: dequantize the output matrix (Q6_K)

_mem_mtime = [0.0]


DIRTY = os.environ.get("FMEM_DIRTY") == "1"          # patched engine: reload only the rows listed in MEM.dirty
ROWSCALE = os.environ.get("FMEM_ROWSCALE_FILE", "")   # patched engine: effective B[u] = rowscale[u] * B[u]


def _replace(tmp, dst):
    for _ in range(50):          # the engine may hold the file open for a moment (Windows)
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(tmp, dst)


def touch_mem(rows=None):
    """the engine reloads memory.bin when its whole-second mtime changes: make every update strictly later, even
    two updates within one second (e.g. an update wipe followed by the new write).
    rows: the B rows changed by this update (dirty-row reload); None = whole file"""
    old = os.stat(MEM).st_mtime
    t = max(time.time(), _mem_mtime[0] + 1.0, old + 1.0)
    _mem_mtime[0] = t
    dp = MEM + ".dirty"
    if not DIRTY or rows is None:
        if os.path.exists(dp):
            os.remove(dp)
    else:
        if isinstance(rows, (set, frozenset)):
            rows = list(rows)
        rs = set(int(r) for r in np.asarray(rows, dtype=np.int64).ravel())
        base = int(old)
        if os.path.exists(dp):
            with open(dp, "rb") as fh:
                b = fh.read()
            if b[:8] == b"STRFDRT1":
                pb, pn, cnt = struct.unpack("<qqi", b[8:28])
                if pn == int(old):           # the previous update was also listed: one list covers both
                    base = pb
                    rs |= set(np.frombuffer(b[28:28 + 4 * cnt], np.int32).tolist())
        arr = np.array(sorted(rs), np.int32)
        with open(dp + ".tmp", "wb") as fh:
            fh.write(b"STRFDRT1" + struct.pack("<qqi", base, int(t), len(arr)) + arr.tobytes())
        _replace(dp + ".tmp", dp)
    os.utime(MEM, (t, t))


def rs_load(D):
    """per-row scales (ones when off or missing)"""
    if ROWSCALE and os.path.exists(ROWSCALE):
        b = np.fromfile(ROWSCALE, np.uint8)
        if b[:8].tobytes() == b"STRFRSC1" and int(np.frombuffer(b[8:12].tobytes(), np.int32)[0]) == D:
            return np.frombuffer(b[12:].tobytes(), np.float32).copy()
    return np.ones(D, np.float32)


def rs_save(rs):
    if not ROWSCALE:
        return
    with open(ROWSCALE + ".tmp", "wb") as fh:
        fh.write(b"STRFRSC1" + struct.pack("<i", len(rs)) + np.asarray(rs, np.float32).tobytes())
    _replace(ROWSCALE + ".tmp", ROWSCALE)

def find_tensor(name):
    sys.path.insert(0, os.path.join(LLM, "tools"))
    from gguf_reader import GGUFFile
    for p in (GGUF, GGUF2):
        f = GGUFFile(p)
        for t in f.tensors:
            if t.name == name:
                return p, f.data_start + t.offset, t
    raise SystemExit(f"{name} not found")


def dequant_q6k(raw, n_blocks):
    """ggml Q6_K: per 256 values ql[128] qh[64] scales[16] (int8) d (fp16) = 210 bytes."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 210)
    ql, qh = b[:, :128].astype(np.int16), b[:, 128:192].astype(np.int16)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)[:, 0]
    y = np.empty((n_blocks, 256), np.float32)
    for h in range(2):
        L, Q, S = ql[:, 64 * h:64 * h + 64], qh[:, 32 * h:32 * h + 32], sc[:, 8 * h:8 * h + 8]
        l = np.arange(32)
        q1 = ((L[:, l] & 0xF) | (((Q[:, l] >> 0) & 3) << 4)) - 32
        q2 = ((L[:, l + 32] & 0xF) | (((Q[:, l] >> 2) & 3) << 4)) - 32
        q3 = ((L[:, l] >> 4) | (((Q[:, l] >> 4) & 3) << 4)) - 32
        q4 = ((L[:, l + 32] >> 4) | (((Q[:, l] >> 6) & 3) << 4)) - 32
        isc = l // 16
        base = 128 * h
        y[:, base + l] = d[:, None] * S[:, isc + 0] * q1
        y[:, base + 32 + l] = d[:, None] * S[:, isc + 2] * q2
        y[:, base + 64 + l] = d[:, None] * S[:, isc + 4] * q3
        y[:, base + 96 + l] = d[:, None] * S[:, isc + 6] * q4
    return y.reshape(-1)


def build_unembed():
    path, off, t = find_tensor("output.weight")
    n_embd, vocab = t.shape
    assert t.type_name == "Q6_K", t.type_name
    row_bytes = n_embd // 256 * 210
    U = np.lib.format.open_memmap(UCACHE, mode="w+", dtype=np.float16, shape=(vocab, n_embd))
    acc = np.zeros(n_embd, np.float64)
    with open(path, "rb") as f:
        f.seek(off)
        for r0 in range(0, vocab, 4096):
            nr = min(4096, vocab - r0)
            rows = dequant_q6k(f.read(nr * row_bytes), nr * n_embd // 256).reshape(nr, n_embd)
            U[r0:r0 + nr] = rows.astype(np.float16)
            acc += rows.sum(0)
    U.flush()
    np.save(UMEAN, (acc / vocab).astype(np.float32))
    print("unembedding", vocab, "x", n_embd)
    return n_embd


# ---------------------------------------------------------------- memory file
HDR = 8 + 16


def mem_layout(n_mem, D, n):
    """byte offsets of each layer's mu, sd, P, B, bias (version 2: a per-unit BCM threshold after B)"""
    off = HDR + 4 * n_mem
    lay = []
    for _ in range(n_mem):
        mu = off; sd = mu + 4 * n; P = sd + 4 * n; B = P + 2 * n * D; bias = B + 2 * D * n
        lay.append((mu, sd, P, B, bias))
        off = bias + 4 * D
    return lay, off


def read_header():
    with open(MEM, "rb") as f:
        assert f.read(8) == b"STRFMEM2"
        n_mem, D, k, n = struct.unpack("<4i", f.read(16))
        layers = list(struct.unpack(f"<{n_mem}i", f.read(4 * n_mem)))
    return n_mem, D, k, n, layers


def init(a):
    n = build_unembed() if not os.path.exists(UCACHE) else np.load(UCACHE, mmap_mode="r").shape[1]
    layers = [int(x) for x in a.layers.split(",")]
    n_mem, D, k = len(layers), a.D, a.k
    lay, size = mem_layout(n_mem, D, n)
    rng = np.random.default_rng(a.seed)
    with open(MEM, "wb") as f:
        f.write(b"STRFMEM2" + struct.pack("<4i", n_mem, D, k, n) + struct.pack(f"<{n_mem}i", *layers))
        for _ in layers:
            f.write(np.zeros(n, np.float32).tobytes())
            f.write(np.ones(n, np.float32).tobytes())
            f.write((rng.standard_normal((n, D), dtype=np.float32) / math.sqrt(n)).astype(np.float16).tobytes())
            f.write(np.zeros((D, n), np.float16).tobytes())
            f.write(np.zeros(D, np.float32).tobytes())
    print("memory.bin", size / 2 ** 20, "MiB; layers", layers, "D", D, "k", k)


# ---------------------------------------------------------------- engine I/O
def chat(messages, max_tokens=1, temperature=0.0):
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False},
            "llm_checkpoint": False}   # one-shot: no conversation checkpoint to restore later
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, e.read()[:300], flush=True)
            if attempt == 3:
                raise
            time.sleep(3)


def read_capture():
    with open(CAP, "rb") as f:
        assert f.read(8) == b"STRFCAP2"
        n_mem, n, resume, nt = struct.unpack("<4i", f.read(16))
        toks = np.frombuffer(f.read(4 * nt), np.int32)
        k, nrec = struct.unpack("<2i", f.read(8))
        rec = np.frombuffer(f.read(4 * nrec * (3 + k)), np.int32).reshape(nrec, 3 + k)
        stats = np.frombuffer(f.read(4 * 2 * n_mem * n), np.float32).reshape(n_mem, 2, n)
    # rows of each memory layer in launch order: (seq, row) sorted
    rows = {}
    for s in range(n_mem):
        r = rec[rec[:, 2] == s]
        r = r[np.lexsort((r[:, 1], r[:, 0]))]
        rows[s] = r[:, 3:]
    return toks, resume, rows, stats


RAW = "[[RAW]]"            # patched server: plain text, no chat template
PRE = "Memory notes.\n"    # fixed preamble: writing and recalling see the same context before the fact


def capture_request_messages(messages):
    """Run one 1-token request with these chat messages (template applied by the server); returns the capture:
    the full prompt's token ids (template included), as the engine positions them."""
    t0 = time.time()
    chat(messages, max_tokens=1)
    for _ in range(100):
        if os.path.exists(CAP) and os.path.getmtime(CAP) >= t0 - 1:
            return read_capture()
        time.sleep(0.05)
    raise SystemExit("no capture file: is the engine running with LLM_FMEM_CAPTURE?")


def capture_request(text):
    """Run one request whose prompt contains `text`; returns the capture after the engine wrote it."""
    t0 = time.time()
    chat([{"role": "user", "content": RAW + PRE + text}], max_tokens=1)
    for _ in range(100):
        if os.path.exists(CAP) and os.path.getmtime(CAP) >= t0 - 1:
            return read_capture()
        time.sleep(0.05)
    raise SystemExit("no capture file: is the engine running with LLM_FMEM_CAPTURE?")


# ---------------------------------------------------------------- tokens -> text (byte-level BPE vocab)
_vocab = None


def id_to_bytes():
    global _vocab
    if _vocab is None:
        v = json.load(open(TOKENIZER, encoding="utf-8"))   # token -> id (byte-level BPE strings)
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        dec = {chr(c): b for b, c in zip(bs, cs)}
        _vocab = {}
        for tok, i in v.items():
            _vocab[i] = bytes(dec.get(ch, 0) for ch in tok)
    return _vocab


def fact_positions(toks, text):
    """token indices whose bytes fall inside the LAST occurrence of `text` in the prompt"""
    vb = id_to_bytes()
    pieces = [vb.get(int(t), b"") for t in toks]
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos); pos += len(p)
    full = b"".join(pieces)
    i = full.rfind(text.encode())
    if i < 0:
        raise SystemExit("fact not found in the prompt tokens")
    j = i + len(text.encode())
    return [t for t, s in enumerate(starts) if s >= i and s < j]


# ---------------------------------------------------------------- calibrate / write
def calibrate(a):
    n_mem, D, k, n, layers = read_header()
    lay, _ = mem_layout(n_mem, D, n)
    sents = [l.strip() for l in open(a.sentences, encoding="utf-8") if l.strip()]
    S = np.zeros((n_mem, 2, n), np.float64)
    cnt = 0
    for s in sents:
        toks, resume, rows, stats = capture_request(s)
        S += stats
        cnt += len(rows[0])
    mu = S[:, 0] / cnt
    sd = np.sqrt(np.maximum(S[:, 1] / cnt - mu ** 2, 1e-4))
    mm = np.memmap(MEM, dtype=np.uint8, mode="r+")
    for l in range(n_mem):
        mm[lay[l][0]:lay[l][0] + 4 * n] = np.frombuffer(mu[l].astype(np.float32).tobytes(), np.uint8)
        mm[lay[l][1]:lay[l][1] + 4 * n] = np.frombuffer(sd[l].astype(np.float32).tobytes(), np.uint8)
    mm.flush(); del mm
    touch_mem()
    print("calibrated on", len(sents), "sentences,", cnt, "rows; mean sd", float(sd.mean()))


def write(a):
    n_mem, D, k, n, layers = read_header()
    lay, _ = mem_layout(n_mem, D, n)
    U = np.load(UCACHE, mmap_mode="r")
    Ubar = np.load(UMEAN)
    toks, resume, rows, _ = capture_request(a.text)
    pos = fact_positions(toks, a.text)
    mm = np.memmap(MEM, dtype=np.uint8, mode="r+")
    Bs = [np.ndarray((D, n), np.float16, mm, lay[l][3]) for l in range(n_mem)]
    w = a.eta / math.sqrt(k)
    used = 0
    for t in pos:
        if t + 1 >= len(toks) or t + 1 > pos[-1] or t < resume:
            continue
        y = int(toks[t + 1])
        d = U[y].astype(np.float32) - Ubar
        d *= a.gain / (np.linalg.norm(d) + 1e-6) / n_mem
        for l in range(n_mem):
            r = t - resume
            if r >= len(rows[l]):
                continue
            u = rows[l][r]
            Bs[l][u] = (Bs[l][u].astype(np.float32) + w * d).astype(np.float16)
        used += 1
    decay = math.exp(-1 / a.tau_w)
    for B in Bs:
        B *= np.float16(decay)
    mm.flush(); del Bs, mm
    touch_mem()
    print(f"wrote {used} positions of: {a.text}")


def next_dist(prompt):
    """the model's own next-token distribution after a raw prompt: top-20 (id, p) from the API"""
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": RAW + prompt}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False}, "llm_checkpoint": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    lpc = (out["choices"][0].get("logprobs") or {}).get("content") or []
    if not lpc:
        # the server leaves logprobs out when the top token is a special one (end of turn): the model wanted to stop
        return []
    lp = lpc[0]["top_logprobs"]
    rev = {b: i for i, b in id_to_bytes().items()}
    res = []
    for e in lp:
        b = bytes(e["bytes"]) if e.get("bytes") is not None else e["token"].encode()
        if b in rev:
            res.append((rev[b], math.exp(e["logprob"])))
    return res


def write_head(a):
    """Head mode (engine LLM_FMEM_HEAD=1): the value lives at the head input, so the exact direction that raises
    log p(y) is u_y - E_p[u].  Per position: target = dl * (1 - p_y) * du / |du|^2, i.e. a logit gain of
    dl * (1 - p_y) nats for y relative to the model's current expectation (Gemini's calibration)."""
    n_mem, D, k, n, layers = read_header()
    lay, _ = mem_layout(n_mem, D, n)
    U = np.load(UCACHE, mmap_mode="r")
    Ubar = np.load(UMEAN)
    toks, resume, rows, _ = capture_request(a.text)
    pos = fact_positions(toks, a.text)
    vb = id_to_bytes()
    w_items = []
    for t in pos:
        if t + 1 > pos[-1] or t < resume:
            continue
        y = int(toks[t + 1])
        prefix = PRE + b"".join(vb.get(int(x), b"") for x in toks[pos[0]:t + 1]).decode("utf-8", "replace")
        dist = next_dist(prefix)
        ps = np.array([p for _, p in dist], np.float32)
        Eu = (ps[:, None] * np.stack([U[i].astype(np.float32) for i, _ in dist])).sum(0) + (1 - ps.sum()) * Ubar
        p_y = dict(dist).get(y, 0.0)
        du = U[y].astype(np.float32) - Eu
        w_items.append((t - resume, a.dl * (1 - p_y) * du / (float(du @ du) + 1e-6), p_y))
    mm = np.memmap(MEM, dtype=np.uint8, mode="r+")
    Bs = [np.ndarray((D, n), np.float16, mm, lay[l][3]) for l in range(n_mem)]
    # recall adds sum over slots of sum_r B[u_r] / sqrt(k); a full match then returns the target exactly
    w = 1.0 / (math.sqrt(k) * n_mem)
    for r, tgt, _ in w_items:
        for l in range(n_mem):
            if r < len(rows[l]):
                u = rows[l][r]
                Bs[l][u] = (Bs[l][u].astype(np.float32) + w * tgt).astype(np.float16)
    decay = math.exp(-1 / a.tau_w)
    for B in Bs:
        B *= np.float16(decay)
    mm.flush(); del Bs, mm
    touch_mem()
    print(f"wrote {len(w_items)} positions (p_y before: {[round(p, 2) for _, _, p in w_items]}) of: {a.text}")


def unit_stats(rows_list, D):
    A = np.concatenate(rows_list)
    cnt = np.bincount(A.ravel(), minlength=D).astype(np.float64)
    srt = np.sort(cnt)[::-1]
    return cnt, {"rows": len(A), "used": int((cnt > 0).sum()), "top1%": round(float(srt[:D // 100].sum() / cnt.sum()), 3),
                 "max_row_share": round(float(srt[0] / len(A)), 3)}


def bcm(a):
    """Iterative BCM-like threshold (from the original theory, 4.2): units that win too often get a higher
    threshold, silent units a lower one, until every unit wins about k/D of the time on ordinary sentences."""
    n_mem, D, k, n, layers = read_header()
    lay, _ = mem_layout(n_mem, D, n)
    sents = [l.strip() for l in open(a.sentences, encoding="utf-8") if l.strip()]
    target = k / D
    for ep in range(a.epochs + 1):
        rows = {l: [] for l in range(n_mem)}
        for snt in sents:
            toks, resume, rr, _ = capture_request(snt)
            p0 = fact_positions(toks, snt)[0] - resume
            for l in range(n_mem):
                rows[l].append(rr[l][p0:len(toks) - resume])
        report = []
        mm = np.memmap(MEM, dtype=np.uint8, mode="r+")
        for l in range(n_mem):
            cnt, st = unit_stats(rows[l], D)
            report.append(st)
            if ep < a.epochs:
                f = cnt / st["rows"]
                b = np.ndarray((D,), np.float32, mm, lay[l][4])
                b -= a.eta * np.log((f + 0.1 * target) / target)
                np.clip(b, -a.clip, a.clip, out=b)
        mm.flush(); del mm
        touch_mem()
        print(f"epoch {ep}: {report}", flush=True)


def reset(a):
    """forget everything: B = 0 (P, mu, sd stay)"""
    n_mem, D, k, n, layers = read_header()
    lay, _ = mem_layout(n_mem, D, n)
    mm = np.memmap(MEM, dtype=np.uint8, mode="r+")
    for l in range(n_mem):
        mm[lay[l][3]:lay[l][3] + 2 * D * n] = 0
    mm.flush(); del mm
    rs_save(np.ones(D, np.float32))
    touch_mem()
    print("memory cleared")


FACTS = [("Leo", "passport", "under the doormat"), ("Mia", "spare key", "inside the red teapot"),
         ("Omar", "watch", "in the bathroom cabinet"), ("Sara", "glasses", "behind the bookshelf"),
         ("Nina", "ticket", "in the blue envelope"), ("Tom", "charger", "in the garage drawer"),
         ("Yumi", "ring", "under the piano"), ("Axel", "wallet", "in the freezer")]


def test(a):
    """baseline -> calibrate -> write facts -> ask; writes test_<tag>.json"""
    out = {"gain": a.gain, "head": a.head, "dl": a.dl, "reverse": a.reverse, "questions": []}
    q = lambda nm, o: f"Where is {nm}'s {o}? Answer in one short sentence."
    generic = "What is the capital of France? Answer in one word."
    out["generic_before"] = chat([{"role": "user", "content": generic}], max_tokens=10)
    base = {f"{nm}/{o}": chat([{"role": "user", "content": q(nm, o)}], max_tokens=40) for nm, o, _ in FACTS}
    if a.calibrate:
        calibrate(argparse.Namespace(sentences=os.path.join(HERE, "calib_sentences.txt")))
    reset(a)
    for nm, o, p in (FACTS[::-1] if a.reverse else FACTS):
        if a.head:
            write_head(argparse.Namespace(text=f"{nm}'s {o} is {p}.", dl=a.dl, tau_w=40.0))
        else:
            write(argparse.Namespace(text=f"{nm}'s {o} is {p}.", eta=1.0, gain=a.gain, tau_w=40.0))
    time.sleep(1)
    hits = hits_cue = 0
    for nm, o, p in FACTS:
        ans = chat([{"role": "user", "content": q(nm, o)}], max_tokens=40)
        # self-cued recall: the thinker starts the sentence itself and memory completes it
        cue = chat([{"role": "user", "content": RAW + PRE + f"{nm}'s {o} is"}], max_tokens=8)
        ok = p.split()[-1].lower() in ans.lower()
        ok_c = p.split()[-1].lower() in cue.lower()
        hits += ok
        hits_cue += ok_c
        out["questions"].append({"fact": f"{nm}'s {o} is {p}.", "before": base[f"{nm}/{o}"], "after": ans,
                                 "correct": ok, "cued": cue, "cued_correct": ok_c})
        print(ok, ok_c, "|", ans, "| cue:", cue)
    out["recall_cued"] = hits_cue / len(FACTS)
    out["generic_after"] = chat([{"role": "user", "content": generic}], max_tokens=10)
    out["recall"] = hits / len(FACTS)
    print("recall", out["recall"], "cued", out["recall_cued"], "| generic:", out["generic_before"], "->", out["generic_after"])
    json.dump(out, open(os.path.join(HERE, f"test_head{a.head}_dl{a.dl}_rev{a.reverse}_{a.tag}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def ask(a):
    print(chat([{"role": "user", "content": a.text}], max_tokens=a.max_tokens))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("init"); i.add_argument("--layers", default="38,43"); i.add_argument("--D", type=int, default=16384)
    i.add_argument("--k", type=int, default=16); i.add_argument("--seed", type=int, default=0)
    c = sp.add_parser("calibrate"); c.add_argument("sentences")
    w = sp.add_parser("write"); w.add_argument("text"); w.add_argument("--eta", type=float, default=1.0)
    w.add_argument("--gain", type=float, default=30.0); w.add_argument("--tau_w", type=float, default=40.0)
    sp.add_parser("reset")
    bc = sp.add_parser("bcm"); bc.add_argument("sentences"); bc.add_argument("--epochs", type=int, default=5)
    bc.add_argument("--eta", type=float, default=0.3); bc.add_argument("--clip", type=float, default=3.0)
    t = sp.add_parser("test"); t.add_argument("--gain", type=float, default=30.0); t.add_argument("--calibrate", type=int, default=1)
    t.add_argument("--reverse", type=int, default=0); t.add_argument("--tag", default="")
    t.add_argument("--head", type=int, default=0); t.add_argument("--dl", type=float, default=10.0)
    q = sp.add_parser("ask"); q.add_argument("text"); q.add_argument("--max_tokens", type=int, default=60)
    a = p.parse_args()
    {"init": init, "calibrate": calibrate, "write": write, "ask": ask, "reset": reset, "test": test, "bcm": bcm}[a.cmd](a)
