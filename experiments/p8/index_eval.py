"""H35 offline (Gemini round 11): can the network do the finding? A hippocampal-style index in fast weights replaces
the external text-embedding search of retrieve-then-read (H28).

Every logged row u (a fact with the sentence it was heard in) gets a fixed random code c(u) in {-1,+1}^Dout / sqrt(Dout).
Its key z(u) is a sparse activation over 65536 units built only from layer-0 word vectors and sound:
  meaning bank (units 0..32767): each token of "entity slot value heard-sentence" -> its layer-0 vector, normalised,
                                 times a fixed random projection -> its top KT units;
  sound bank (32768..65535):     every toneless pinyin syllable and syllable pair (confusable initials and finals
                                 merged, as in the memory key) -> KS hashed units.
Units are weighted by 1/sqrt(n_i + 10), n_i = how many rows have used unit i (rare units count more), the K strongest
are kept, and z is normalised. Write (delta rule): W += eta * z (c - gamma * W^T z)^T. Read: y = W^T z(q) for the heard
question, scores = c(u) . y over the rows logged before the question, top 5 go to the reader.

Layer-0 vectors here are the input token embeddings of Qwen3-Embedding-0.6B (same tokenizer family as LLM server; the
deployed version would use LLM server's own token embeddings, as the memory key already does). The comparison is recall@5:
of the told questions whose answer exists in the log before the question, how many have a matching row in the top 5,
for the network index and for the dense text-embedding search used in E43.

    python index_eval.py <zh_result_*.json> <timeline.json> [...]
env: IX_DOUT (256), IX_K (128), IX_KT (4), IX_KS (2), IX_GAMMA (0.2)
"""
import hashlib
import json
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from label_finder_eval import overlap  # noqa: E402
from rescore_norm import norm, UNKNOWN  # noqa: E402
from read_loss import READ_INSTR, row_text  # noqa: E402

D, H = 65536, 32768
DOUT = int(os.environ.get("IX_DOUT", "256"))
K = int(os.environ.get("IX_K", "128"))
KT = int(os.environ.get("IX_KT", "4"))
KS = int(os.environ.get("IX_KS", "2"))
GAMMA = float(os.environ.get("IX_GAMMA", "0.2"))
CONF = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]
SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}


def h32(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


class Keys:
    def __init__(self, enc):
        rng = np.random.default_rng(1)
        if os.environ.get("IX_EMB") == "llm":
            # LLM server's own tokenizer and layer-0 word vectors (rows for the tokens these logs use, llm_tok/)
            from transformers import Qwen2TokenizerFast
            sd = os.path.join(HERE, "llm_tok")
            self.tok = Qwen2TokenizerFast(vocab_file=os.path.join(sd, "vocab.json"),
                                          merges_file=os.path.join(sd, "merges.txt"))
            ids = np.load(os.path.join(sd, "ids.npy"))
            rows = np.load(os.path.join(sd, "emb_rows.npy")).astype(np.float32)
            self.E = {int(i): r for i, r in zip(ids, rows)}
            sample = np.load(os.path.join(sd, "emb_sample.npy")).astype(np.float32)
            dim = rows.shape[1]
        else:
            self.tok = enc.tokenizer
            E = enc[0].auto_model.get_input_embeddings().weight.detach().float().numpy()
            self.E = E
            sample = E[np.sort(rng.integers(0, E.shape[0], 4000))]
            dim = E.shape[1]
        self._init_proj(sample, dim, rng)

    def _init_proj(self, sample, dim, rng):
        sample = (sample - sample.mean(1, keepdims=True)) / (sample.std(1, keepdims=True) + 1e-5)
        self.mu, self.sd = sample.mean(0), sample.std(0) + 1e-3
        self.P = (rng.standard_normal((dim, H)) / np.sqrt(dim)).astype(np.float32)
        self.cache = {}

    def token_units(self, t):
        u = self.cache.get(t)
        if u is None and isinstance(self.E, dict) and t not in self.E:
            u = np.zeros(0, np.int64)          # a token without a copied word vector: its sound units still count
            self.cache[t] = u
        if u is None:
            x = self.E[t]
            x = ((x - x.mean()) / (x.std() + 1e-5) - self.mu) / self.sd
            u = np.argpartition(-(x @ self.P), KT)[:KT]
            self.cache[t] = u
        return u

    def raw(self, text):
        """unit -> count (before frequency weighting)"""
        from pypinyin import lazy_pinyin, Style
        cnt = {}
        for t in self.tok(text, add_special_tokens=False).input_ids:
            piece = self.tok.decode([t]).strip()
            if not piece or not re.search(r"[\w一-鿿]", piece):
                continue
            for u in self.token_units(t):
                cnt[int(u)] = cnt.get(int(u), 0) + 1
        syl = []
        for ch in text:
            if "一" <= ch <= "鿿":
                s = lazy_pinyin(ch, style=Style.NORMAL)[0]
                for a, b in CONF:
                    s = s.replace(a, b)
                syl.append(s)
        feats = [f"u:{s}" for s in syl] + [f"b:{a}_{b}" for a, b in zip(syl, syl[1:])]
        for ft in feats:
            for j in range(KS):
                u = H + h32(f"{ft}#{j}") % H
                cnt[u] = cnt.get(u, 0) + 1
        return cnt


SPLIT = os.environ.get("IX_SPLIT") == "1"          # H38a (Gemini round 12): K/2 meaning units + K/2 sound units
UNLEARN = float(os.environ.get("IX_UNLEARN", "0"))  # H38b: targeted unlearning strength alpha (0 = off)
UTH = float(os.environ.get("IX_UTH", "0.3"))        # H38b: key overlap (cosine) that counts as "the same thing again"


def _topk_norm(units, w, k):
    if len(units) > k:
        keep = np.argpartition(-w, k)[:k]
        units, w = units[keep], w[keep]
    return units, w / (np.linalg.norm(w) + 1e-9)


def sparse(cnt, n_use):
    units = np.array(list(cnt), np.int64)
    w = np.array([cnt[u] for u in units], np.float32) / np.sqrt(n_use[units] + 10.0)
    if not SPLIT:
        return _topk_norm(units, w, K)
    m = units < H
    u1, w1 = _topk_norm(units[m], w[m], K // 2)
    u2, w2 = _topk_norm(units[~m], w[~m], K // 2)
    return np.concatenate([u1, u2]), np.concatenate([w1, w2]) * np.float32(np.sqrt(0.5))


def write_row(W, C, n_use, keys_store, written, row, cnt):
    """delta-rule write of one row; with IX_UNLEARN, first weaken the shared units -> code links of the existing rows
    whose key overlaps this one (cosine >= UTH) among the 3 the index ranks highest for this key (H38b)"""
    n_use[list(cnt)] += 1
    u, w = sparse(cnt, n_use)
    if UNLEARN > 0 and written:
        y = w @ W[u]
        cand = np.array(written)
        sc = C[cand] @ y
        new = dict(zip(u.tolist(), w.tolist()))
        for j in cand[np.argsort(-sc)[:3]]:
            uo, wo = keys_store[int(j)]
            old = dict(zip(uo.tolist(), wo.tolist()))
            shared = [x for x in new if x in old]
            if not shared:
                continue
            cos = sum(new[x] * old[x] for x in shared)
            if cos < UTH:
                continue
            us = np.array(shared, np.int64)
            ws = np.array([min(new[x], old[x]) for x in shared], np.float32)
            ws /= np.linalg.norm(ws) + 1e-9
            W[us] -= UNLEARN * np.outer(ws, C[int(j)])
    W[u] += np.outer(w, C[row] - GAMMA * (w @ W[u]))
    keys_store[row] = (u, w)
    written.append(row)


def run(path, tl_path, enc, keys):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(tl_path, encoding="utf-8"))}
    log = [{"label": (w["fact"][0], w["fact"][1]), "value": str(w["fact"][2]), "day": w["day"], "start": w["start"],
            "src": w["src"], "heard": w.get("heard") or ""} for w in d["writes"]]
    # rows enter the index in time order (night rows at the end of their day)
    order = sorted(range(len(log)), key=lambda i: (log[i]["day"], log[i]["src"] == "sleep", log[i]["start"]))
    rng = np.random.default_rng(int(os.environ.get("IX_SEED", "7")))   # row-code seed
    C = (rng.choice(np.array([-1.0, 1.0], np.float32), (len(log), DOUT)) / np.sqrt(DOUT)).astype(np.float32)
    if os.environ.get("IX_ORTHO") == "1":
        # H41: the first DOUT rows get exactly orthogonal codes (rows of a random orthogonal matrix), so a row's score is
        # its key overlap with the question and nothing else; rows beyond DOUT keep random +-1 codes
        q_, _ = np.linalg.qr(rng.standard_normal((DOUT, DOUT)))
        n_o = min(len(log), DOUT)
        C[order[:n_o]] = q_[:n_o].astype(np.float32)
    R = enc.encode([row_text(ev) for ev in log], normalize_embeddings=True, batch_size=32)
    raws = [keys.raw(f"{ev['label'][0]} {SLOT_ZH.get(ev['label'][1], ev['label'][1])} {ev['value']} {ev['heard']}")
            for ev in log]
    qs, q_span = [], []
    for r in d["questions"]:
        q = r["q"]
        if q["expect"] is None and not READ:
            continue
        seg = next((g for g in tl[q["day"]]["segments"] if g["text"] == q["text"]), None)
        if seg is None:
            continue
        cand = [a for a in d["answers"] if a["day"] == q["day"] and
                min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        if not cand:
            continue
        a = max(cand, key=lambda a: overlap(q["entity"], (a["anchor"] or "").split("|")[0]))
        qs.append((q, seg, a.get("heard") or q["text"]))
        q_span.append((a["start"], a["end"]))
    # replay time: write rows and answer questions in order
    W = np.zeros((D, DOUT), np.float32)
    n_use = np.zeros(D, np.float32)
    written, keys_store = [], {}
    events = [("w", (log[i]["day"], 1 if log[i]["src"] == "sleep" else 0, log[i]["start"]), i) for i in order]
    events += [("q", (q["day"], 0, seg["start"]), j) for j, (q, seg, _) in enumerate(qs)]
    if SUPERSEDE:                           # H42: after the night's rows, the sleep marks superseded older rows
        events += [("s", (dd, 2, 0.0), dd) for dd in sorted({ev["day"] for ev in log})]
    events.sort(key=lambda x: (x[1], 0 if x[0] in ("w", "s") else 1))
    superseded, sup_log = set(), []
    res = {"net": 0, "dense": 0, "either": 0, "n": 0}
    misses, strength, dump = [], [], []
    read_res = {"ok": 0, "wrong": 0, "none": 0, "never_answered": 0, "never_kept": 0}
    for kind, _, j in events:
        if kind == "w":
            i = j
            write_row(W, C, n_use, keys_store, written, i, raws[i])
            continue
        if kind == "s":
            sleep_supersede(j, log, raws, written, superseded, sup_log, W, C, n_use)
            continue
        q, seg, heard = qs[j]
        avail = [i for i in written if (log[i]["day"] < q["day"] or (log[i]["src"] != "sleep" and log[i]["start"] < seg["start"]))
                 and i not in superseded]
        if not avail:
            continue
        told = q["expect"] is not None
        alts = nbest_alts(q["day"], q_span[j], heard)
        cnt = keys.raw(heard)
        for alt in alts:                    # H39b: the other recognition hypotheses add their units at weight 0.3
            for k_, v_ in keys.raw(alt).items():
                cnt[k_] = cnt.get(k_, 0) + 0.3 * v_
        u, w = sparse(cnt, n_use)
        y = w @ W[u]
        s = C[avail] @ y
        top5 = [avail[k] for k in np.argsort(-s)[:5]]
        net_top = set(top5)
        match = {i for i in avail if told and overlap(norm(q["expect"]), norm(log[i]["value"])) >= 0.6}
        strength.append(("told" if match else ("never" if not told else "told_absent"), float(s.max()),
                         float(np.linalg.norm(y))))
        if READ:
            outcome = read_one(q, heard, top5, log, alts)
            read_res[outcome] += 1
            # H37 (ask-back after "I don't know"): what would asking about the index's first row give?
            best = top5[0]
            dump.append({"day": q["day"], "told": told, "outcome": outcome, "top_score": float(s.max()),
                         "second": float(np.sort(s)[-2]) if len(s) > 1 else 0.0,
                         "top_label": list(log[best]["label"]), "top_value": log[best]["value"],
                         "top_right": bool(told and overlap(norm(q["expect"]), norm(log[best]["value"])) >= 0.6),
                         "heard": heard})
        if not match:
            continue
        res["n"] += 1
        qv = enc.encode(READ_INSTR + heard, normalize_embeddings=True)
        dense_top = {avail[k] for k in np.argsort(-(R[avail] @ qv))[:5]}
        hn, hd = bool(net_top & match), bool(dense_top & match)
        res["net"] += hn
        res["dense"] += hd
        res["either"] += hn or hd
        if hd and not hn:
            best = min(match, key=lambda i: -float(C[i] @ y))
            misses.append((q["day"], heard[:30], row_text(log[best])[:50]))
    print("==", os.path.basename(path), f"Dout {DOUT} K {K} KT {KT} KS {KS} gamma {GAMMA} | answerable told {res['n']} | "
          f"recall@5 network index {res['net']} | dense text embedding {res['dense']} | either {res['either']}",
          flush=True)
    for m in misses:
        print("    dense only:", m)
    # can the readout strength alone say "never heard" before any reading (Gemini round 11, pre-retrieval gate)?
    for f_i, name in ((1, "top score"), (2, "|y|")):
        a_ = [x[f_i] for x in strength if x[0] == "told"]
        b_ = [x[f_i] for x in strength if x[0] == "never"]
        if a_ and b_:
            auc = sum((x > z) + 0.5 * (x == z) for x in a_ for z in b_) / (len(a_) * len(b_))
            print(f"  readout {name}: told-with-answer median {np.median(a_):.3f}, never-told median {np.median(b_):.3f}, "
                  f"AUC {auc:.2f}", flush=True)
    if SUPERSEDE:
        print(f"  sleep supersede: {len(superseded)} rows marked superseded", flush=True)
        for x in sup_log:
            print("    ", x, flush=True)
    if READ and os.environ.get("IX_DUMP"):
        json.dump(dump, open(os.environ["IX_DUMP"] + "_" + os.path.basename(path), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    if READ:
        print(f"  read end to end (network top 5 + LLM server reader + sound-twin second read): right {read_res['ok']} | "
              f"wrong {read_res['wrong']} | none {read_res['none']} || never-told answered {read_res['never_answered']} "
              f"/ {read_res['never_answered'] + read_res['never_kept']}", flush=True)


READ = os.environ.get("IX_READ") == "1"


SUPERSEDE = os.environ.get("IX_SUPERSEDE") == "1"
SUP_SYS = ("你是居家助理 Aria 的記憶整理員，現在是睡前。下面是一筆今天的新紀錄，和幾筆比較舊的紀錄（都附當時聽到的原話，原話可能有語音辨識錯字）。"
           "判斷哪些舊紀錄已經被新紀錄取代：必須是同一個東西、同一方面，而新紀錄給了新的內容（改口、換位置、改時間、狀態變了、借出去了）。"
           "只是相關、不同方面、不同的東西、或新舊內容其實一樣，都不算取代。\n"
           "輸出一行：被取代的舊紀錄編號（用逗號分隔），或「無」。")


SUP_SYS_STRICT = ("你是居家助理 Aria 的記憶整理員，現在是睡前。下面是一筆今天的新紀錄，和幾筆比較舊的紀錄（都附當時聽到的原話，原話可能有語音辨識錯字）。"
                  "判斷哪些舊紀錄已經被新紀錄取代。三個條件都要成立才算：\n"
                  "1. 是同一個實體（不是同一類的另一樣東西：「舊的」和「新的」、兩個不同的人、兩種不同的藥都算不同實體）；\n"
                  "2. 是同一方面（放在哪、時間、狀態……要相同）；\n"
                  "3. 新紀錄的內容和舊紀錄不同，而且是在更新它（改口、換位置、改時間、狀態變了、借出去了）。\n"
                  "只要有一個條件不確定，就不算。\n"
                  "輸出一行：被取代的舊紀錄編號（用逗號分隔），或「無」。")


def CLEAN_OK(v):
    from value_clean import clean
    return clean(v)[1]


def sleep_supersede(day, log, raws, written, superseded, sup_log, W, C, n_use):
    """H42 (Gemini round 11, sleep consolidation): every row logged today looks up the 3 older rows the index ranks
    highest for it; the thinker judges which of them the new row supersedes; those are left out of later answering
    (the event log itself is not changed)."""
    import re as _re
    from retrieve_read_eval import chat, row_text as rt
    older = [i for i in written if log[i]["day"] < day and i not in superseded]
    if not older:
        return
    strict = os.environ.get("IX_SUP_STRICT") == "1"
    for i in [i for i in written if log[i]["day"] == day]:
        if strict and not CLEAN_OK(log[i]["value"]):
            continue                         # H42b: a listening row that is not a fact supersedes nothing
        u, w = sparse(raws[i], n_use)
        y = w @ W[u]
        sc = C[older] @ y
        cand = [older[k] for k in np.argsort(-sc)[:3]]
        if strict:                           # H42b: only the same aspect can be superseded
            cand = [c for c in cand if log[c]["label"][1] == log[i]["label"][1]]
            if not cand:
                continue
        lines = [f"{n + 1}. 第{log[c]['day']}天｜{rt(log[c])}" for n, c in enumerate(cand)]
        user = f"今天是第{day}天。\n新紀錄：第{day}天｜{rt(log[i])}\n舊紀錄：\n" + "\n".join(lines)
        out = chat([{"role": "system", "content": SUP_SYS_STRICT if strict else SUP_SYS},
                    {"role": "user", "content": user}], max_tokens=20)
        first = out.strip().split("\n")[0]
        if "無" in first:
            continue
        for m in _re.findall(r"\d+", first):
            k = int(m)
            if 1 <= k <= len(cand) and cand[k - 1] not in superseded:
                superseded.add(cand[k - 1])
                sup_log.append((day, rt(log[i])[:40], "supersedes", rt(log[cand[k - 1]])[:40]))


NBEST = json.load(open(os.environ["IX_NBEST"], encoding="utf-8")) if os.environ.get("IX_NBEST") else None
NBEST_READ = os.environ.get("IX_NBEST_READ") == "1"


def nbest_alts(day, span, heard):
    """H39b: recognition hypotheses 2-3 of the heard segments overlapping the question's answer span (distinct texts)"""
    if not NBEST:
        return []
    out = []
    for x in NBEST:
        if x["day"] == day and min(x["end"], span[1]) - max(x["start"], span[0]) > 0.3:
            for h in x["nbest"]:
                t = h["text"]
                if t and t != heard and t not in out and t not in heard:
                    out.append(t)
    return out[:3]


def read_one(q, heard, top5, log, alts=()):
    """the E43 reader (retrieve_read_eval SYS, sound-twin second read) on the network's top 5"""
    import re as _re
    from retrieve_read_eval import SYS, chat, row_text as rt, sound_twins
    top = sorted(top5, key=lambda i: (log[i]["day"], log[i]["start"]))
    lines = [f"{n + 1}. 第{log[i]['day']}天｜{rt(log[i])}" for n, i in enumerate(top)]
    out = chat([{"role": "system", "content": SYS},
                {"role": "user", "content": f"今天是第{q['day']}天。\n使用者問：{heard}\n記憶紀錄：\n" + "\n".join(lines)}])
    m = _re.search(r"編號\s*(\d+)", out)
    pick = top[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= len(top) else None
    if pick is None or UNKNOWN.search(out.split("\n")[0]):
        pick = None
        twins = sound_twins(heard, [(i, log[i]["label"][0]) for i in top])
        if twins:
            lines2 = [f"{n + 1}. 第{log[i]['day']}天｜{rt(log[i])}" for n, (i, _) in enumerate(twins)]
            note = "；".join(f"問句裡的「{sub}」和紀錄裡的「{log[i]['label'][0]}」唸起來一樣" for i, sub in twins)
            out = chat([{"role": "system", "content": SYS},
                        {"role": "user", "content": f"今天是第{q['day']}天。\n使用者問：{heard}\n（注意：{note}，是語音辨識聽錯，"
                                                    f"指的是同一個東西。還是要問的那一方面對得上才回答。）\n記憶紀錄：\n" + "\n".join(lines2)}])
            m = _re.search(r"編號\s*(\d+)", out)
            pick = twins[int(m.group(1)) - 1][0] if m and 1 <= int(m.group(1)) <= len(twins) else None
            if pick is not None and UNKNOWN.search(out.split("\n")[0]):
                pick = None
    told = q["expect"] is not None
    if pick is None:
        return "none" if told else "never_kept"
    if not told:
        return "never_answered"
    reply = out.split("\n", 1)[1].strip() if "\n" in out else ""
    return "ok" if overlap(norm(q["expect"]), norm(log[pick]["value"] + " " + reply)) >= 0.6 else "wrong"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    keys = Keys(enc)
    a = sys.argv[1:]
    for p, t in zip(a[0::2], a[1::2]):
        run(p, t, enc, keys)
