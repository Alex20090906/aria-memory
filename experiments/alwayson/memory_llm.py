"""E41: the simulator's memory on LLM server's real in-weights memory (AI PC, memory service on port 8094 via SSH tunnel),
in the best long-term setup found so far:

- the network is only the familiarity gate: key echo (FMEM_ECHO=1 on the service), no decay, delta-rule writes;
- content comes from an append-only event log looked up by exact label (U15);
- optional: when no exact label exists, the semantic label finder (Qwen3-Embedding service, port 8095) offers the most
  similar heard label as an ask-back, and only if the network also finds that label familiar (E37, H22).

Same interface as memory_local.Brain where the simulator uses it. Sleep does nothing to the network (no decay, so no
consolidation is needed); it is logged with its real duration so the always-on loop's sleep timing stays measurable.
The thinker (chat) is LLM server on port 8093, set by the simulator.
"""
import json
import re
import time
import urllib.request

import numpy as np

SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
FIND_INSTR = "Instruct: 給一句使用者的問話，找出它在問記憶裡的哪一件事（東西加上哪一方面）\nQuery: "

# E44: retrieve-then-read (H28) with the sound-twin second read (H31b), the same prompts as zh_v3 (E42/E43).
READ_K = 5
READ_INSTR = "Instruct: 給一句使用者的問話，找出記憶裡在講同一件事的紀錄\nQuery: "
READ_SYS = ("你是居家助理 Aria，只服務一位使用者。使用者剛問了你一句話（語音辨識結果，可能有錯字）。下面是你記憶裡和這句話最接近的幾筆紀錄，"
            "每筆有：第幾天、東西、哪一方面、內容、當時聽到的原話（也可能有錯字）。這幾筆可能全都不相關。\n"
            "規則：\n"
            "1. 只有某一筆明確是在講使用者問的那一件事（同一個東西、同一方面），才用它回答。\n"
            "2. 只是類似的東西（例如問太陽眼鏡、紀錄是備用眼鏡；問個人網銀、紀錄是企業網銀）不算，回答不知道。\n"
            "3. 同一件事有好幾筆，以天數最新的為準。\n"
            "輸出兩行：第一行「編號 N」或「不知道」；第二行用一句話回答使用者（不知道就說不記得）。")
UNKNOWN = re.compile(r"(不記得|不知道|沒有印象|沒聽你說|沒講過|沒說過|忘了|不確定|don't|do not)")
_CONF = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]


def row_text(r):
    e, s = r["label"]
    return f"{e}｜{SLOT_ZH.get(s, s)}｜{r['value']}（原話：{r['heard']}）"


def _syl(ch):
    from pypinyin import lazy_pinyin, Style
    s = lazy_pinyin(ch, style=Style.NORMAL)[0]
    for a, b in _CONF:
        s = s.replace(a, b)
    return s


def sound_twins(question, rows):
    """[(row index, word in the question)] for rows whose name sounds like a different word in the question (H31b)."""
    qc = [(ch, _syl(ch)) for ch in question if "一" <= ch <= "鿿"]
    out = []
    for i, e in rows:
        if len(e) < 2 or not all("一" <= ch <= "鿿" for ch in e):
            continue
        es = [_syl(ch) for ch in e]
        n = len(es)
        for a in range(len(qc) - n + 1):
            win = qc[a:a + n]
            diff = sum(x[1] != y for x, y in zip(win, es))
            sub = "".join(x[0] for x in win)
            if sub != e and (diff == 0 or (n >= 4 and diff == 1)):
                out.append((i, sub))
                break
    return out


def _post(url, obj, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(obj, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if i == 4:
                raise
            time.sleep(3 * (i + 1))


class LLMMemory:
    def __init__(self, url="http://127.0.0.1:8094", D=65536, find="", tau=0.6, embed_url="http://127.0.0.1:8095/embed",
                 chat_url="http://127.0.0.1:8093/v1/chat/completions"):
        self.url, self.D, self.find, self.tau, self.embed_url = url, D, find, tau, embed_url
        self.chat_url = chat_url
        self.gate, self.pmin = 0.5, 0.7
        self.c = np.zeros(D, np.int32)
        self.prov, self.elog, self.ring = {}, [], []
        self.writes = 0
        self.log_mode = True
        self.timing = {"gen_tok_s": [], "prefill_s": []}
        self.calls = {"write_s": [], "familiar_s": []}
        self._fam_cache, self._keyvec = {}, {}
        self.last_mode = "direct"
        self._tok = None
        self._queue, self._pending = [], {}
        self.rows, self._rowvec = [], {}          # E44: every logged fact with its day and heard sentence (H28)
        self.calls["read_s"] = []
        self.calls["embed_rows_s"] = []
        self.embed_idle = False                   # H34 (E45): embed rows during write-behind idle time
        self.index = None                         # H35 (E46): NetIndex, the network finds the rows
        self.gate_on = True                       # E47: False skips the familiarity check on the picked row
        r = _post(self.url + "/reset", {})
        assert r.get("ok"), r

    # ---------------------------------------------------------------- network
    def _familiar(self, e, s):
        if (e, s) in self._pending:            # heard, network write still queued (H26)
            return 1.0, self._familiar_raw(e, s)[1]
        return self._familiar_raw(e, s)

    def _familiar_raw(self, e, s):
        if (e, s) in self._fam_cache:          # valid until the next write (writes clear the cache)
            return self._fam_cache[(e, s)]
        t0 = time.time()
        r = _post(self.url + "/familiar", {"e": e, "s": s})
        self.calls["familiar_s"].append(time.time() - t0)
        out = (float(r["fam"]), np.array(r["units"], np.int64))
        self._fam_cache[(e, s)] = out
        return out

    def calibrate(self, pairs):
        pass

    def anchor_units(self, e, s):
        return self._familiar(e, s)[1]

    def familiarity(self, e, s):
        return self._familiar(e, s)[0]

    def write(self, e, s, v, src="owner", decay=True, heard="", day=0):
        t0 = time.time()
        _post(self.url + "/write2", {"e": e, "s": s, "v": v})
        self.calls["write_s"].append(time.time() - t0)
        self._fam_cache.clear()
        self.ring.append((e, s))
        self.ring = self.ring[-64:]
        self.prov[(e, s)] = {"src": src, "value": v, "at_write": self.writes}
        self.elog.append(((e, s), v))
        self.rows.append({"label": (e, s), "value": v, "day": day, "heard": heard or "(sleep)"})
        self._index_row()
        self.writes += 1

    # ---------------------------------------------------------------- write-behind (H26)
    def enqueue(self, e, s, v, src="owner", heard="", day=0):
        """The fact goes into the event log at once (content is answerable now); the network write waits for idle time.
        A label still waiting counts as heard: the gate exists to say "this was heard", and it was."""
        self._queue.append((e, s, v, src))
        self._pending[(e, s)] = self._pending.get((e, s), 0) + 1
        self.prov[(e, s)] = {"src": src, "value": v, "at_write": self.writes}
        self.elog.append(((e, s), v))
        self.rows.append({"label": (e, s), "value": v, "day": day, "heard": heard or "(sleep)"})
        self._index_row()
        self.ring.append((e, s))
        self.ring = self.ring[-64:]

    def pending_writes(self):
        return len(self._queue)

    def flush_one(self):
        e, s, v, src = self._queue.pop(0)
        t0 = time.time()
        _post(self.url + "/write2", {"e": e, "s": s, "v": v})
        dt = time.time() - t0
        self.calls["write_s"].append(dt)
        self._fam_cache.clear()
        self._pending[(e, s)] -= 1
        if not self._pending[(e, s)]:
            del self._pending[(e, s)]
        self.writes += 1
        if self.embed_idle and self.index is None:
            self.embed_rows()          # H34: row vectors are made in idle time, not when the owner asks
        return time.time() - t0            # network write (+ row embedding with embed_idle), all in idle time

    def use_index(self):
        from net_index import NetIndex
        self.index = NetIndex()
        self.calls["index_q_s"], self.calls["index_w_s"] = [], []

    def _index_row(self):
        """the newest row goes into the network index at once (a few milliseconds)"""
        if self.index is not None:
            r = self.rows[-1]
            t0 = time.time()
            self.index.write(f"{r['label'][0]} {SLOT_ZH.get(r['label'][1], r['label'][1])} {r['value']} {r['heard']}")
            self.calls["index_w_s"].append(time.time() - t0)

    def embed_rows(self):
        """Embed every logged row that has no vector yet (H34). E44 did this at question time: up to 19 rows, 2-4 s."""
        new = [i for i in range(len(self.rows)) if i not in self._rowvec]
        if new:
            t0 = time.time()
            for i, v in zip(new, self._embed([row_text(self.rows[i]) for i in new])):
                self._rowvec[i] = np.asarray(v, np.float32)
            self.calls["embed_rows_s"].append(time.time() - t0)

    # ---------------------------------------------------------------- recall
    def _embed(self, texts):
        return _post(self.embed_url, {"texts": texts})["vectors"]

    def _find_label(self, question, e, s):
        latest = {}
        for k, v in self.elog:
            latest[k] = v
        labs = [l for l in latest if l != (e, s)]
        if not labs:
            return None
        new = [l for l in labs if l not in self._keyvec]
        if new:
            for l, vec in zip(new, self._embed([f"{l[0]}（{SLOT_ZH.get(l[1], l[1])}）" for l in new])):
                self._keyvec[l] = np.asarray(vec, np.float32)
        qv = np.asarray(self._embed([FIND_INSTR + f"{question}（{e}，{SLOT_ZH.get(s, s)}）"])[0], np.float32)
        sims = np.stack([self._keyvec[l] for l in labs]) @ qv
        j = int(np.argmax(sims))
        return labs[j], latest[labs[j]], float(sims[j])

    def recall(self, e, s, question="", n_new=16, bump=True, gated=True):
        """(answer, familiarity, confidence, passed). last_mode says how: direct, ask (ask-back on a similar label), none."""
        self.last_mode = "none"
        fam, anc = self._familiar(e, s)
        val = next((v for k, v in reversed(self.elog) if k == (e, s)), "")
        if val and (fam >= self.gate or not gated):
            if bump:
                self.c[anc] += 1
            self.last_mode = "direct"
            return val, fam, 1.0, True
        if not gated:
            return val, fam, 0.0, False
        if self.find == "embed" and question and self.elog:
            hit = self._find_label(question, e, s)
            if hit and hit[2] >= self.tau:
                (e2, s2), v2, cos = hit
                fam2, anc2 = self._familiar(e2, s2)
                if fam2 >= self.gate:
                    if bump:
                        self.c[anc2] += 1
                    self.last_mode = "ask"
                    self.last_ask = {"label": f"{e2}|{s2}", "cos": round(cos, 3), "fam": round(fam2, 3)}
                    return v2, fam2, 1.0, True
        return "", fam, 0.0, False

    # ---------------------------------------------------------------- retrieve-then-read (H28, H31b; same as zh_v3)
    def _chat(self, system, user, max_tokens=80):
        body = {"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "system", "content": system},
                                                                  {"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False},
                "llm_checkpoint": False}
        r = _post(self.chat_url, body, timeout=600)
        return (r["choices"][0]["message"].get("content") or "").strip().splitlines()

    def read(self, question, day, hint=None, phon=False):
        """(value, reply, familiarity, ok, label). The heard question retrieves the k nearest logged facts; the thinker
        names the one about exactly what was asked or says it does not know; the network is the gate of the picked fact."""
        t0 = time.time()
        self.last_mode = "read"
        out = self._read(question, day, hint, phon)
        self.calls["read_s"].append(time.time() - t0)
        return out

    def _read(self, question, day, hint, phon):
        if not self.rows:
            return "", "我不記得了。", 0.0, False, None
        if self.index is not None:          # H35 (E46): the network index finds the rows, no text embedding
            t0 = time.time()
            top = sorted(self.index.query(question, READ_K), key=lambda i: (self.rows[i]["day"], i))
            self.calls["index_q_s"].append(time.time() - t0)
        else:
            self.embed_rows()               # E44: all rows not yet embedded; with embed_idle only the just-heard ones
            qv = np.asarray(self._embed([READ_INSTR + question])[0], np.float32)
            sims = np.stack([self._rowvec[i] for i in range(len(self.rows))]) @ qv
            top = sorted(np.argsort(-sims)[:READ_K].tolist(), key=lambda i: (self.rows[i]["day"], i))
        lines = [f"{n + 1}. 第{self.rows[i]['day']}天｜{row_text(self.rows[i])}" for n, i in enumerate(top)]
        user = (f"今天是第{day}天。\n使用者問：{question}\n" + (f"（這一題問的重點可能是：{hint}，可能聽錯）\n" if hint else "")
                + "記憶紀錄：\n" + "\n".join(lines))
        out = self._chat(READ_SYS, user)
        first = out[0] if out else ""
        m = re.search(r"編號\s*(\d+)", first)
        reply = " ".join(out[1:]).strip()
        pick = top[int(m.group(1)) - 1] if m and not UNKNOWN.search(first) and 1 <= int(m.group(1)) <= len(top) else None
        if pick is None and phon:
            twins = sound_twins(question, [(i, self.rows[i]["label"][0]) for i in top])
            if twins:
                note = "；".join(f"問句裡的「{sub}」和紀錄裡的「{self.rows[i]['label'][0]}」唸起來一樣" for i, sub in twins)
                lines = [f"{n + 1}. 第{self.rows[i]['day']}天｜{row_text(self.rows[i])}" for n, (i, _) in enumerate(twins)]
                user = (f"今天是第{day}天。\n使用者問：{question}\n（注意：{note}，是語音辨識聽錯，指的是同一個東西。"
                        f"還是要問的那一方面對得上才回答。）\n記憶紀錄：\n" + "\n".join(lines))
                out = self._chat(READ_SYS, user)
                first = out[0] if out else ""
                m = re.search(r"編號\s*(\d+)", first)
                reply = " ".join(out[1:]).strip()
                ok_ = m and not UNKNOWN.search(first) and 1 <= int(m.group(1)) <= len(twins)
                pick = twins[int(m.group(1)) - 1][0] if ok_ else None
                self.last_mode = "read_phon"
        if pick is None:
            return "", reply or "我不記得了。", 0.0, False, None
        e, s = self.rows[pick]["label"]
        if not self.gate_on:
            # E47: the gate on the picked row is skipped. 508 reads in E42-E46 were never vetoed, because every logged
            # row is written to the network (or queued, which counts as heard): the gate passes by construction.
            return self.rows[pick]["value"], reply, 1.0, True, (e, s)
        fam, anc = self._familiar(e, s)
        if fam < self.gate:
            self.last_mode = "read_gated"            # the network never heard it
            return "", "我不記得了。", fam, False, (e, s)
        self.c[anc] += 1
        return self.rows[pick]["value"], reply, fam, True, (e, s)

    def sleep(self, last_window):
        return {"replayed": 0, "candidates": 0, "rejected": 0, "consolidated_units": 0, "downscaled_units": 0,
                "note": "no decay: nothing to consolidate"}

    def participation_ratio(self):
        return 0.0          # not measured on the real memory (H23: not a vegetative-state measure)

    def n_tokens(self, s):
        # same tokenizer the other runs used for heard text (memory_local.MODEL), so token shares stay comparable
        if self._tok is None:
            from transformers import AutoTokenizer
            from memory_local import MODEL
            self._tok = AutoTokenizer.from_pretrained(MODEL)
        return len(self._tok(s or "").input_ids)
