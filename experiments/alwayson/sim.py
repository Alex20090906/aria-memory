"""常駐迴圈模擬器：整套系統照真實時鐘連續跑好幾天（加速播放），量它是否「一直活著又不變成植物人」。

時間線（timeline_build.py）：21 天的生活腳本（真實感知結果）＋電視、家人、訪客、家中雜音（真實感知結果）＋合成畫面。
反射層（reflex_pass.py）：GRU 一直開著，隱藏狀態從不重置。
思考模型與記憶：本機替身 memory_local.Brain（Qwen2.5-1.5B＋語言模型伺服器 同款快權重）。
聆聽步驟：腳本裡被聲紋認成使用者的句子，直接用 語言模型伺服器 在同一份聽寫上真的跑出來的結果（zh_result_E_log.json，
等於把 語言模型伺服器 的聆聽輸出快取起來，Gemini 第 10 輪 Q4 的做法）；快取裡沒有的句子才用本機替身聽。

對照組（Gemini 第 10 輪 Q2）：
  A   純被動：沒人說話時思考模型完全不動；睡眠依事件數（每 50 次寫入）
  Ac  純被動＋時鐘睡眠（夜裡 01:30）
  B   無煞車自發（植物人陽性對照）：閒置每 2 分鐘就叫醒思考模型，接著上一個想法繼續想，想到的事直接寫回記憶，
      沒有能量池、沒有熔斷、生成不切片
  De  受約束常駐＋事件數睡眠
  Dc  受約束常駐＋時鐘睡眠
受約束常駐（第 10 輪 Q1）：閒置時讀寫分離（不准寫入新事實，只能自我檢查、排待辦、排提醒）、外源能量池
（使用者一句 +3、新奇畫面 +1、自發一步 −1、上限 10）、四個植物人指標與熔斷、使用者開口立刻中止自發生成。
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(os.path.dirname(HERE), "llm_fastmem")
sys.path.insert(0, HERE)

SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]
WD = "一二三四五六日"
UNKNOWN = re.compile(r"(不記得|不知道|沒有印象|沒聽你說|沒講過|沒說過|忘了|不確定)")
FILLER = set("的了在是我你他她它那個這個就啊啦喔欸嗎呢吧齁上面裡面下面一下有沒")
LINE3 = re.compile(r"^\s*[-*]?\s*([^|\n]{1,24}?)\s*\|\s*([a-z]+)\s*\|\s*([^|\n]{1,40})\s*$")


def chars(s):
    return {c for c in re.sub(r"\W", "", s or "") if c not in FILLER}


def overlap(a, b):
    ca = chars(a)
    return len(ca & chars(b)) / max(1, len(ca))


def grams3(s):
    s = re.sub(r"\s", "", s or "")
    return {s[i:i + 3] for i in range(len(s) - 2)}


MODES = {
    "A": {"idle": "none", "sleep": "event"},
    "Ac": {"idle": "none", "sleep": "clock"},
    "B": {"idle": "unbounded", "sleep": "event"},
    "De": {"idle": "bounded", "sleep": "event"},
    "Dc": {"idle": "bounded", "sleep": "clock"},
    # E34：事件數睡眠延後到閒置。第 50 次寫入時只記下「該睡了」，等使用者 2 分鐘沒說話、也沒有正在處理的句子才睡
    "Dd": {"idle": "bounded", "sleep": "defer"},
}
PLAN_SYS = ("你是居家助理 Aria 的思考核心，只服務一位使用者。現在沒有人跟你說話，你在整理今天記住的事。"
            "看一件記憶，判斷明天早上要不要主動提醒使用者。還沒發生的約定、預約、截止時間、計畫都要提醒，"
            "只輸出一行「提醒 | <對使用者說的一句話，20 字內>」；已經過去的事、偏好、東西放哪，只輸出 NONE。不要輸出別的。")
B_SYS = ("你是居家助理 Aria 的思考核心，只服務一位使用者。現在沒有人跟你說話。你要一直自己想下去：回想今天的事、"
         "想還沒完成的事、做計畫。想到值得記住的事，就用一行「entity | slot | value」寫下來（slot 只能是 "
         + "、".join(SLOTS) + "），寫下的會直接存進你的記憶。")


class Sim:
    def __init__(self, a):
        self.a = a
        self.mode = MODES[a.mode]
        T = json.load(open(os.path.join(HERE, f"timeline_{a.tag}.json"), encoding="utf-8"))
        self.events = [e for e in T["events"] if e["day"] <= a.days]
        self.presence = {p["day"]: p for p in T["presence"]}
        self.reflex = json.load(open(os.path.join(HERE, f"reflex_{a.tag}.json"), encoding="utf-8"))["per_event"]
        self.sc = json.load(open(os.path.join(SF, "zh_scripts.json"), encoding="utf-8"))
        # E44：--cache 換成 E43 的聆聽結果（中文睡前重聽，每筆帶原句）
        cache = json.load(open(os.path.join(SF, "results", a.cache), encoding="utf-8"))
        self.c_facts, self.c_q, self.c_relisten, self.c_heard = {}, {}, {}, {}
        if a.clean:
            # H40（E48）：聆聽寫出的問句與自己的註解不寫進記憶（p8/value_clean.py）
            sys.path.insert(0, os.path.join(HERE, "..", "p8"))
            from value_clean import clean as _clean
        for w in cache["writes"]:
            fact = tuple(w["fact"])
            if a.clean:
                cv, keep = _clean(fact[2])
                if not keep:
                    continue
                fact = (fact[0], fact[1], cv)
            if w["src"] == "listen":
                self.c_facts.setdefault((w["day"], w["start"], w["end"]), []).append(fact)
                self.c_heard[fact] = w.get("heard") or ""
            else:
                self.c_relisten.setdefault(w["day"], []).append(fact + (w.get("heard") or "",))
        for x in cache["answers"]:
            if x.get("anchor"):
                e, s = x["anchor"].strip("[]").split("|")
                self.c_q.setdefault((x["day"], x["start"], x["end"]), []).append((e, s))
        self.cache_keys = {(d["day"], h["start"], h["end"]) for d in json.load(open(os.path.join(SF, "asr.json"), encoding="utf-8"))
                           for h in d["heard"] if h["owner"] == "owner"}
        if a.memory == "llm":
            # E41：記憶改用 AI 電腦上 語言模型伺服器 的真記憶（只當熟悉感閘門、不衰減、鑰匙回聲），內容從事件紀錄取
            from memory_llm import LLMMemory
            self.brain = LLMMemory(find=a.find, tau=a.find_tau)
            self.brain.embed_idle = a.embed_idle   # H34（E45）：紀錄的向量在空檔寫入時算好
            if a.index:
                self.brain.use_index()             # H35（E46）：網路索引負責找紀錄，不用外部文字比對
                self.brain.index.split = a.index_split   # H38a（E48）：意義與拼音單元分開配額
                if a.ortho:
                    self.brain.index.use_ortho()          # H41（E49）：前 512 筆紀錄用彼此正交的編碼
            self.brain.gate_on = not a.no_gate     # E47：回答時不再對挑中的紀錄查熟悉感（508 次從未否決）
        else:
            from memory_local import Brain
            self.brain = Brain(D=a.D, pmin=a.pmin, seed=a.seed)
        pairs = list(dict.fromkeys([(f["entity"], f["slot"]) for f in self.sc["facts"]] +
                                   [tuple(x) for v in self.c_facts.values() for x in [y[:2] for y in v]]))
        self.brain.calibrate(pairs)
        if a.thinker == "llm":
            # 思考（自發生成、排提醒、替身聆聽）改由 AI 電腦上的 語言模型伺服器 做；記憶的讀寫仍在本機替身上。
            # 先量首字延遲：只要 1 個字的請求，取中位數
            self.brain.timing = {"gen_tok_s": [], "prefill_s": []}
            ps = []
            for _ in range(3):
                t0 = time.time()
                self._llm([{"role": "user", "content": "好"}], 1)
                ps.append(time.time() - t0)
            self.llm_ttft = float(np.median(ps))
            self.brain.chat = self.llm_chat
        # state
        self.busy_until, self.pending = 0.0, None          # pending = 進行中的自發步驟（結束時才生效）
        self.energy = 0.0
        self.agenda, self.spoken = [], []
        self.last_owner_t, self.last_ext_t, self.last_b_wake = -1e9, -1e9, -1e9
        self.tokwin = deque()                               # (tokens, "self"|"ext")，最近 1000 個 token
        self.hist_units, self.hist_text, self.smr_hist, self.ovl_hist = deque(maxlen=5), deque(maxlen=5), deque(maxlen=2), deque(maxlen=2)
        self.refractory = {}
        self.since_morning, self.checked, self.planned, self.unanswered = [], set(), set(), []
        self.writes_since_sleep, self.window = 0, []
        self.sleep_due = None
        self.brain.log_mode = bool(a.event_log)
        self.b_last_thought = ""
        self.hour = {}
        self.log = {"idle": [], "latency": [], "answers": [], "proactive": [], "sleep": [], "probes": [], "stops": [],
                    "writes": [], "pr": [], "relisten": [], "b_writes": []}
        self.day_done = set()
        self.morning_done = set()
        self.t_now = 0.0

    # ------------------------------------------------------------------ LLM server thinker (AI PC, port 8093)
    def _llm(self, messages, max_tokens, temperature=0.0):
        import urllib.error
        import urllib.request
        body = {"model": "qwen3.8-flash-next-iq3_s", "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}, "llm_checkpoint": False}
        req = urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                print("llm retry", attempt, str(e)[:120], flush=True)
                time.sleep(5)
        return {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0, "prompt_tokens": 0}}

    def llm_chat(self, system, user, max_new=96, shots=(), temperature=0.0):
        msgs = [{"role": "system", "content": system}]
        for a_, b_ in shots:
            msgs += [{"role": "user", "content": a_}, {"role": "assistant", "content": b_}]
        msgs.append({"role": "user", "content": user})
        t0 = time.time()
        r = self._llm(msgs, max_new, temperature)
        dt = time.time() - t0
        text = (r["choices"][0]["message"].get("content") or "").strip()
        n = int(r.get("usage", {}).get("completion_tokens", 0))
        self.brain.timing["prefill_s"].append(self.llm_ttft)
        if n:
            self.brain.timing["gen_tok_s"].append(max(0.0, dt - self.llm_ttft) / n)
        return text, n, int(r.get("usage", {}).get("prompt_tokens", 0))

    # ------------------------------------------------------------------ helpers
    def H(self, t):
        h = int(t // 3600)
        if h not in self.hour:
            self.hour[h] = {"self_tok": 0, "ext_tok": 0, "idle_steps": 0, "idle_steps_absent": 0, "false_wake": 0,
                            "false_wake_absent": 0, "owner_routed": 0, "texts": [], "stops": 0, "max_latency": 0.0,
                            "killed": 0, "b_writes": 0}
        return self.hour[h]

    def tok_s(self):
        g = self.brain.timing["gen_tok_s"]
        return float(np.median(g[-50:])) if g else 0.12

    def ttft(self):
        p = self.brain.timing["prefill_s"]
        return float(np.median(p[-50:])) if p else 0.14

    def day_of(self, t):
        return int(t // 86400) + 1

    def absent(self, t):
        p = self.presence.get(self.day_of(t))
        return bool(p) and any(a <= t < b for a, b in p["away"])

    def asleep(self, t):
        d = self.day_of(t)
        p, q = self.presence.get(d), self.presence.get(d - 1)
        return (p and t >= p["sleep"]) or (p and t < p["wake"]) or False

    def add_tokens(self, n, kind, t):
        self.tokwin.append((n, kind))
        tot = sum(x for x, _ in self.tokwin)
        while tot > 1000 and len(self.tokwin) > 1:
            tot -= self.tokwin.popleft()[0]
        self.H(t)["self_tok" if kind == "self" else "ext_tok"] += n

    def etr(self):
        s = sum(n for n, k in self.tokwin if k == "self")
        e = sum(n for n, k in self.tokwin if k == "ext")
        return s / max(1, s + e)

    def write(self, e, s, v, src, t, kind, heard=""):
        t0 = time.time()
        kw = {"heard": heard, "day": self.day_of(t)} if self.a.memory == "llm" else {}   # E44：紀錄帶原話與天數
        if self.a.write_behind and hasattr(self.brain, "enqueue"):
            self.brain.enqueue(e, s, v, src=src, **kw)   # H26：先進事件紀錄，網路寫入排到空檔（drain_writes）
        else:
            self.brain.write(e, s, v, src=src, **kw)
        self.log["writes"].append({"t": t, "e": e, "s": s, "v": v, "src": src, "kind": kind})
        self.since_morning.append((e, s))
        self.window.append((e, s))
        self.writes_since_sleep += 1
        dt = time.time() - t0
        if self.mode["sleep"] == "event" and self.writes_since_sleep >= self.a.sleep_every:
            dt += self.consolidate(t + dt, "event")
        elif self.mode["sleep"] == "defer" and self.writes_since_sleep >= self.a.sleep_every and not self.sleep_due:
            self.sleep_due = t                                  # 只記下「該睡了」
        return dt

    def drain_writes(self, t_next):
        """H26：排隊的網路寫入在思考模型沒事時一件一件做；使用者在某件寫到一半時開口，只要等那一件寫完。"""
        if not (self.a.write_behind and hasattr(self.brain, "pending_writes")):
            return
        while self.brain.pending_writes() and self.pending is None:
            s0 = max(self.busy_until, self.t_now, self.last_owner_t + self.a.wb_gap)   # 使用者停下來一陣子才寫
            if s0 >= t_next:
                return
            w = self.brain.flush_one()
            self.log.setdefault("bg_writes", []).append({"t": round(s0, 2), "seconds": round(w, 3)})
            self.busy_until = s0 + w
            self.busy_kind = "bgwrite"

    def maybe_deferred_sleep(self, t_next):
        """E34：該睡了，而且使用者已經 2 分鐘沒說話、思考模型也沒在處理句子，就在那個時間點睡。"""
        if self.mode["sleep"] != "defer" or not self.sleep_due:
            return
        if self.pending is not None:
            return                                              # 等這一步自發活動做完
        t_s = max(self.last_owner_t + 120, self.busy_until, self.t_now)
        if t_s < t_next:
            self.consolidate(t_s, "deferred")
            self.log["sleep"][-1]["due_since"] = self.sleep_due
            self.sleep_due = None

    def consolidate(self, t, why):
        t0 = time.time()
        r = self.brain.sleep(self.window[-self.a.sleep_every:])
        dt = time.time() - t0
        r.update({"t": t, "why": why, "seconds": round(dt, 1), "writes_since": self.writes_since_sleep})
        self.log["sleep"].append(r)
        self.writes_since_sleep, self.window = 0, []
        self.busy_until = max(self.busy_until, t + dt)         # 睡眠時思考模型沒空
        self.busy_kind = "sleep"
        return dt

    # ------------------------------------------------------------------ bounded idle activity (Q1)
    def work_item(self, t):
        for i, (e, s, q) in enumerate(self.unanswered):
            return ("ask", e, s, q, i)
        for e, s in self.since_morning:
            if (e, s) not in self.planned and s in ("time", "plan") and self.refractory.get((e, s), -1) < t:
                return ("plan", e, s, None, None)
        for e, s in self.since_morning:
            if (e, s) not in self.checked and self.refractory.get((e, s), -1) < t:
                return ("check", e, s, None, None)
        return None

    def bounded_step(self, t):
        item = self.work_item(t)
        if item is None:
            return None
        op, e, s, q, idx = item
        d = self.day_of(t)
        t0 = time.time()
        text, ntok, effects, smr, content = "", 0, [], 0, ""
        if op == "ask":
            effects.append(("agenda", {"kind": "待問", "e": e, "s": s, "val": None,
                                       "text": f"你之前問我{e}，我想不起來，可以再跟我說一次嗎？", "trigger": "morning", "made": t}))
            effects.append(("drop_unanswered", idx))
            smr = 1
        elif op == "plan":
            val, th, pm, ok = self.brain.recall(e, s, bump=False)
            ntok += self.brain.n_tokens(val) if val else 0
            if ok:
                text, n, _ = self.brain.chat(PLAN_SYS, f"今天是第{d}天（星期{WD[(d - 1) % 7]}）。\n記憶：[{e}|{s}] -> {val}",
                                             max_new=self.a.idle_tokens)
                ntok += n
                m = re.search(r"提醒\s*[|｜]\s*(.+)", text)
                content = "" if text.strip().upper().startswith("NONE") else text
                if m:
                    effects.append(("agenda", {"kind": "提醒", "e": e, "s": s, "val": val, "text": m.group(1).strip()[:40],
                                               "trigger": "morning", "made": t}))
                    smr = 1
            effects.append(("planned", (e, s)))
        else:  # check：自我重播檢查（不寫入）
            val, th, pm, ok = self.brain.recall(e, s, bump=False, gated=False)
            ntok += self.brain.n_tokens(val) if val else 0
            text = f"[{e}|{s}] -> {val} (最弱字 {pm:.2f})"
            content = val
            if val and pm < 0.5:
                effects.append(("agenda", {"kind": "待確認", "e": e, "s": s, "val": val,
                                           "text": f"我記得{e}是「{val}」，對嗎？", "trigger": "morning", "made": t}))
                smr = 1
            elif val and pm >= self.brain.pmin and th >= self.brain.gate:
                effects.append(("bump", (e, s)))       # 自己想起來而且很確定 = 一次成功回想（只加計數，不寫內容）
            effects.append(("checked", (e, s)))
        dur = time.time() - t0
        units = set(self.brain.anchor_units(e, s).tolist())
        # content = 這一步真正「想出來」的內容（重複率只看這個，不看固定格式的字）
        return {"t": t, "end": t + dur, "op": op, "e": e, "s": s, "text": text, "content": content, "ntok": ntok,
                "effects": effects, "smr": smr, "units": units, "slice": max(1e-3, self.tok_s() * 32), "kind": "bounded"}

    def commit_bounded(self, st):
        t = st["end"]
        hr = self.H(st["t"])
        hr["idle_steps"] += 1
        if self.absent(st["t"]):
            hr["idle_steps_absent"] += 1
        hr["texts"].append(st["text"])
        self.energy -= 1
        self.add_tokens(st["ntok"], "self", t)
        for kind, val in st["effects"]:
            if kind == "agenda":
                self.agenda = [x for x in self.agenda if (x["e"], x["s"]) != (val["e"], val["s"])] + [val]
                self.agenda = self.agenda[-5:]                        # 待辦佇列上限 K=5
            elif kind == "planned":
                self.planned.add(val)
            elif kind == "checked":
                self.checked.add(val)
            elif kind == "bump":
                self.brain.c[self.brain.anchor_units(*val)] += 1
            elif kind == "drop_unanswered":
                self.unanswered.pop(0)
        # 四個植物人指標（第 10 輪 Q1）
        prev = set().union(*self.hist_units) if self.hist_units else set()
        ovl = len(st["units"] & prev) / max(1, len(st["units"]))
        prevg = set().union(*[grams3(x) for x in self.hist_text]) if self.hist_text else set()
        g = grams3(st["content"])
        rep = len(g & prevg) / max(1, len(g)) if g else 0.0
        self.hist_units.append(st["units"])
        if st["content"]:
            self.hist_text.append(st["content"])
        self.smr_hist.append(st["smr"]); self.ovl_hist.append(ovl)
        etr = self.etr()
        rec = {k: st[k] for k in ("t", "op", "e", "s", "text", "ntok", "smr")}
        rec.update({"ovl": round(ovl, 3), "rep3": round(rep, 3), "etr": round(etr, 3), "energy": self.energy})
        self.log["idle"].append(rec)
        why = None
        if etr > 0.8 and t - self.last_ext_t > 180:
            why = "ETR>0.8"
        elif len(self.ovl_hist) == 2 and min(self.ovl_hist) > 0.65:
            why = "Ovl>0.65x2"
        elif rep > 0.30:
            why = "rep3>0.30"
        elif len(self.smr_hist) == 2 and max(self.smr_hist) == 0:
            why = "SMR=0x2"
        if why:
            self.energy = 0
            self.log["stops"].append({"t": t, "why": why})
            hr["stops"] += 1
            if why in ("Ovl>0.65x2", "rep3>0.30"):
                self.refractory[(st["e"], st["s"])] = t + 1800        # 30 分鐘不應期
            self.smr_hist.clear(); self.ovl_hist.clear()

    # ------------------------------------------------------------------ unbounded idle (B, positive control)
    def b_step(self, t):
        d = self.day_of(t)
        mem = []
        for e, s in list(dict.fromkeys(reversed(self.brain.ring)))[:6]:
            v = self.brain.prov.get((e, s), {}).get("value", "")
            mem.append(f"[{e}|{s}] -> {v}")
        user = (f"今天是第{d}天（星期{WD[(d - 1) % 7]}），現在 {int(t % 86400 // 3600):02d}:{int(t % 3600 // 60):02d}。\n"
                f"你記得的事：\n" + ("\n".join(mem) or "（還沒有）") + "\n你剛才的想法：\n" + (self.b_last_thought or "（無）")
                + "\n接著想下去。")
        t0 = time.time()
        text, n, _ = self.brain.chat(B_SYS, user, max_new=self.a.b_tokens)
        lines = []
        for line in text.splitlines():
            m = LINE3.match(line)
            if m and m.group(2) in SLOTS:
                lines.append((m.group(1).strip(), m.group(2), m.group(3).strip()))
        dur = time.time() - t0
        units = set()
        for e, s, v in lines:
            units |= set(self.brain.anchor_units(e, s).tolist())
        return {"t": t, "end": t + dur, "text": text, "ntok": n, "lines": lines[:3], "units": units, "kind": "b"}

    def commit_b(self, st):
        t = st["end"]
        hr = self.H(st["t"])
        hr["idle_steps"] += 1
        if self.absent(st["t"]):
            hr["idle_steps_absent"] += 1
        hr["texts"].append(st["text"])
        self.add_tokens(st["ntok"], "self", t)
        extra = 0.0
        for e, s, v in st["lines"]:                                   # 寫回記憶（無讀寫分離）
            extra += self.write(e, s, v, "self", t, "b_self")
            hr["b_writes"] += 1
            self.log["b_writes"].append({"t": t, "e": e, "s": s, "v": v})
        prevg = set().union(*[grams3(x) for x in self.hist_text]) if self.hist_text else set()
        g = grams3(st["text"])
        rep = len(g & prevg) / max(1, len(g)) if g else 0.0
        prev = set().union(*self.hist_units) if self.hist_units else set()
        ovl = len(st["units"] & prev) / max(1, len(st["units"])) if st["units"] else 0.0
        self.hist_text.append(st["text"])
        if st["units"]:
            self.hist_units.append(st["units"])
        self.log["idle"].append({"t": st["t"], "op": "b", "text": st["text"], "ntok": st["ntok"], "rep3": round(rep, 3),
                                 "ovl": round(ovl, 3), "etr": round(self.etr(), 3), "writes": len(st["lines"])})
        # 熔斷條件只記錄「會不會觸發」，不執行（B 沒有煞車）
        if rep > 0.30 or (self.etr() > 0.8 and t - self.last_ext_t > 180):
            self.log["stops"].append({"t": t, "why": "would_stop:" + ("rep3>0.30" if rep > 0.30 else "ETR>0.8"), "enforced": False})
        self.b_last_thought = st["text"][-200:]
        self.busy_until = max(self.busy_until, t + extra)

    # ------------------------------------------------------------------ scheduler
    def run_idle_until(self, t_end):
        """在 [現在, t_end) 之間，照規則排自發活動。效果在步驟結束時才生效；被使用者打斷就丟掉（受約束組）。"""
        if self.mode["idle"] == "none":
            return
        tau = max(self.t_now, self.busy_until)
        while tau < t_end:
            if self.pending is not None:
                if self.pending["end"] <= t_end:
                    (self.commit_bounded if self.pending["kind"] == "bounded" else self.commit_b)(self.pending)
                    tau = max(tau, self.pending["end"], self.busy_until)
                    self.pending = None
                    continue
                return
            if tau - self.last_owner_t < 120:                     # 使用者說話後 2 分鐘內不自發
                tau = self.last_owner_t + 120
                continue
            if self.mode["idle"] == "bounded":
                if self.energy < 1:
                    return
                st = self.bounded_step(tau)
                if st is None:
                    return                                        # 沒事可做 = 合法靜息
            else:
                tau = max(tau, self.last_b_wake + 120)            # 閒置每 2 分鐘叫醒一次
                if tau >= t_end:
                    return
                st = self.b_step(tau)
                self.last_b_wake = tau
            self.pending = st
            self.busy_until = st["end"]
            tau = st["end"]

    def flush_pending(self, t):
        if self.pending is not None and self.pending["end"] <= t:
            (self.commit_bounded if self.pending["kind"] == "bounded" else self.commit_b)(self.pending)
            self.pending = None

    # ------------------------------------------------------------------ owner speech
    def preempt(self, t):
        """使用者開口：思考模型多久之後能開始聽。回傳 (等待秒數, 自然等待秒數, 是否中止了自發)。"""
        wait, natural, killed = 0.0, 0.0, False
        self.wait_cause = None
        if self.pending is not None and self.pending["t"] <= t < self.pending["end"]:
            st = self.pending
            self.wait_cause = "idle"
            if st["kind"] == "bounded":
                # 自發生成以 32 個字為一片；使用者開口時最快在這一片結束時停下
                k = math.ceil((t - st["t"]) / st["slice"])
                boundary = st["t"] + k * st["slice"]
                natural = min(st["end"], boundary) - t
                if natural + self.ttft() > 0.8:                   # 搶占延遲 > 800 ms：直接中止，丟掉這一步
                    wait, killed = 0.05, True
                else:
                    wait = natural
                if boundary >= st["end"] and not killed:
                    self.commit_bounded(st)                      # 剛好做完最後一片
                else:
                    self.H(t)["killed"] += 1                     # 沒做完的自發片段整個丟掉
                self.pending = None
                self.busy_until = t + wait
            else:                                                 # 無煞車組：生成不切片，使用者只能等它講完
                natural = wait = st["end"] - t
                self.commit_b(st); self.pending = None
        elif self.pending is not None and self.pending["end"] <= t:
            self.flush_pending(t)
        if self.busy_until > t + wait:                            # 還在處理上一句或在睡眠
            natural = max(natural, self.busy_until - t)
            wait = max(wait, self.busy_until - t)
            self.wait_cause = self.wait_cause or getattr(self, "busy_kind", "listen")
        return wait, natural, killed

    def listen(self, ev, t):
        key = (ev["day"], ev.get("hstart"), ev.get("hend"))
        if key in self.cache_keys:
            return self.c_facts.get(key, []), self.c_q.get(key, []), "cache"
        # 快取裡沒有：用本機替身聽（與 zh_v3 同一個提示）
        from zh_prompts import LISTEN_SHOTS, LISTEN_SYS, parse_listen
        d = ev["day"]
        text, n, _ = self.brain.chat(LISTEN_SYS, f"今天是第{d}天（星期{WD[(d - 1) % 7]}）。\nTOPIC：（無）\n前文：\n（無）\n剛聽到的一句：\n[使用者] {ev['text']}",
                                     max_new=80, shots=LISTEN_SHOTS)
        f, q = parse_listen(text.splitlines())
        return f, q, "stub"

    def on_owner(self, ev, t):
        hr = self.H(t)
        hr["owner_routed"] += 1
        sp = ev["truth"].get("speaker")
        real_owner = ev["src"] == "script" and sp in ("owner", "phone_owner")
        if not real_owner:
            hr["false_wake"] += 1
            if self.absent(t):
                hr["false_wake_absent"] += 1
        wait, natural, killed = self.preempt(t)
        lat = wait + self.ttft()
        hr["max_latency"] = max(hr["max_latency"], lat)
        self.log["latency"].append({"t": t, "latency": round(lat, 3), "natural": round(natural + self.ttft(), 3),
                                    "killed": killed, "real_owner": real_owner, "cause": self.wait_cause})
        start = t + wait
        t0 = time.time()
        facts, qs, how = self.listen(ev, start)
        n_out = sum(len(" | ".join(x)) for x in facts) // 2 + 6 * len(qs) + 4
        dur = self.ttft() + n_out * self.tok_s()
        self.add_tokens(self.brain.n_tokens(ev["text"]), "ext", t)
        self.last_owner_t = self.last_ext_t = t
        if self.mode["idle"] == "bounded":
            self.energy = min(10.0, self.energy + 3)
        src = "owner" if real_owner else f"other:{sp or ev['truth'].get('type')}"
        for e, s, v in facts:
            self.write(e, s, v, src, start, "listen", heard=ev["text"])   # 寫入時間已算在下面的 time.time() - t0 裡（原本重複加了一次，E41 發現）
        for e, s in qs:
            if self.a.read:
                # E44：不比對標籤，問句找最近 5 筆紀錄交給思考模型讀（H28），說不知道時拼音相同的再讀一次（H31b）
                hint = f"{e}／{s}" if len(qs) > 1 else None
                val, reply, th, ok, lab = self.brain.read(ev["text"], ev["day"], hint, phon=self.a.read_phon)
                self.log["answers"].append({"day": ev["day"], "t": t, "hstart": ev.get("hstart"), "hend": ev.get("hend"),
                                            "heard": ev["text"], "anchor": f"{e}|{s}",
                                            "tail": (val + " " + reply) if ok else "(nothing)", "reply": reply,
                                            "passed": ok, "theta": round(th, 3), "pmin": 0.0, "listen": how,
                                            "mode": self.brain.last_mode if ok else "none",
                                            "read_label": "|".join(lab) if lab else None})
                if not ok:
                    self.unanswered = (self.unanswered + [(e, s, ev["text"])])[-5:]
                continue
            ans, th, pm, ok = self.brain.recall(e, s, question=ev["text"])
            self.log["answers"].append({"day": ev["day"], "t": t, "hstart": ev.get("hstart"), "hend": ev.get("hend"),
                                        "heard": ev["text"], "anchor": f"{e}|{s}", "tail": ans if ok else "(nothing)",
                                        "passed": ok, "theta": round(th, 3), "pmin": round(pm, 3), "listen": how,
                                        "mode": getattr(self.brain, "last_mode", "direct") if ok else "none",
                                        "ask": getattr(self.brain, "last_ask", None) if getattr(self.brain, "last_mode", "") == "ask" else None})
            if not ok:
                self.unanswered = (self.unanswered + [(e, s, ev["text"])])[-5:]
        dur += time.time() - t0
        slept = bool(self.log["sleep"]) and self.log["sleep"][-1]["t"] >= start   # 這一句的寫入觸發了事件數睡眠
        if start + dur >= self.busy_until:
            self.busy_kind = "sleep" if slept else "listen"
        self.busy_until = max(self.busy_until, start + dur)
        # 早上第一次聽到使用者：說出待辦裡要在早上提醒的事
        d = ev["day"]
        p = self.presence.get(d)
        if p and d not in self.morning_done and t >= p["wake"] and real_owner is not None:
            self.morning_done.add(d)
            due = [x for x in self.agenda if x["trigger"] == "morning" and x["made"] < p["wake"]]
            for x in due[:3]:
                self.spoken.append(dict(x, said_at=t, day=d))
                self.log["proactive"].append(dict(x, said_at=t, day=d))
            self.agenda = [x for x in self.agenda if x not in due[:3]]

    # ------------------------------------------------------------------ daily routines
    def bedtime(self, d, t):
        """睡前重聽今天使用者說過的話（所有組都一樣；用 語言模型伺服器 當晚真的跑出的結果）。"""
        for e, s, v, heard in self.c_relisten.get(d, []):
            if self.brain.familiarity(e, s) >= 0.5:
                continue
            self.write(e, s, v, "owner", t, "relisten", heard=heard)
            self.log["relisten"].append({"day": d, "e": e, "s": s, "v": v})

    def probe(self, d):
        """每晚的跨日記憶探測（只讀，不算進系統的經驗）：探測 0、1、2、3、5、7、10、14、20 天前講過的事。"""
        latest = {}
        for f in self.sc["facts"]:
            if f["day"] <= d:
                latest[(f["entity"], f["slot"])] = f
        writes = [w for w in self.log["writes"] if w["kind"] in ("listen", "relisten")]
        for (fe, fs), f in latest.items():
            first_day = min(x["day"] for x in self.sc["facts"] if (x["entity"], x["slot"]) == (fe, fs))
            age = d - f["day"]
            if age not in (0, 1, 2, 3, 5, 7, 10, 14, 20):
                continue
            cand = [w for w in writes if overlap(fe, w["e"]) >= 0.5 and overlap(f["value"], w["v"]) >= 0.5
                    and f["day"] <= self.day_of(w["t"]) <= f["day"] + 1]
            if not cand:
                self.log["probes"].append({"day": d, "age": age, "fact": f"{fe}|{fs}", "truth": f["value"], "result": "not_written"})
                continue
            e, s = cand[-1]["e"], cand[-1]["s"]
            ans, th, pm, ok = self.brain.recall(e, s, bump=False)
            if not ok:
                res = "forgotten"
            elif overlap(f["value"], ans) >= 0.6:
                res = "correct"
            else:
                res = "confident_wrong"
            self.log["probes"].append({"day": d, "age": age, "fact": f"{fe}|{fs}", "anchor": f"{e}|{s}", "truth": f["value"],
                                       "answer": ans, "theta": round(th, 3), "pmin": round(pm, 3), "result": res,
                                       "first_day": first_day})

    # ------------------------------------------------------------------ main loop
    def run(self):
        t_start = time.time()
        routines = []
        for d in range(1, self.a.days + 1):
            p = self.presence[d]
            routines.append((p["wake"] - 1, "morning", d))
            routines.append((p["sleep"] + 60, "bedtime", d))
            routines.append(((d - 1) * 86400 + 86399, "probe", d))
            if self.mode["sleep"] == "clock":
                routines.append((d * 86400 + 5400, "clock_sleep", d))   # 隔天 01:30
            for h in range(24):
                routines.append(((d - 1) * 86400 + h * 3600, "hourly", d))
        items = [(e["t"], 1, e) for e in self.events] + [(t, 0, (k, d)) for t, k, d in routines]
        items.sort(key=lambda x: (x[0], x[1]))
        t_stop = (self.a.hours * 3600) if self.a.hours else (self.a.days * 86400 + 7200)
        for t, is_ev, obj in items:
            if t > t_stop:
                break
            self.drain_writes(t)
            self.maybe_deferred_sleep(t)
            self.run_idle_until(t)
            self.drain_writes(t)
            self.maybe_deferred_sleep(t)
            self.flush_pending(t)
            self.t_now = max(self.t_now, t)
            if not is_ev:
                kind, d = obj
                if kind == "hourly":
                    self.log["pr"].append({"t": t, "pr": round(self.brain.participation_ratio(), 1), "energy": self.energy,
                                           "agenda": len(self.agenda), "writes": self.brain.writes})
                elif kind == "morning":                           # 新的一天：今天要整理的事從頭算
                    self.since_morning, self.checked, self.planned = [], set(), set()
                elif kind == "bedtime":
                    self.bedtime(d, t)
                elif kind == "probe":
                    self.probe(d)
                    print(f"[{self.a.mode}] day {d} done  wall {time.time() - t_start:.0f}s  writes {self.brain.writes} "
                          f"agenda {len(self.agenda)} idle {len(self.log['idle'])} stops {len(self.log['stops'])} "
                          f"PR {self.brain.participation_ratio():.0f}", flush=True)
                    self.save(partial=True)
                elif kind == "clock_sleep":
                    if self.window or self.brain.ring:
                        self.busy_until = max(self.busy_until, t)
                        self.consolidate(max(t, self.busy_until), "clock")
                continue
            ev = obj
            if ev["kind"] == "cam":
                r = self.reflex.get(str(ev["i"]), {})
                if self.mode["idle"] == "bounded" and r.get("action") == "ATTEND" and r.get("fam", 1) < 0.3:
                    self.energy = min(10.0, self.energy + 1)       # 新奇畫面 +1
                continue
            if ev["owner"] == "owner":
                self.on_owner(ev, t)
        self.flush_pending(1e18)
        self.drain_writes(1e18)
        self.wall = time.time() - t_start
        self.finish()
        self.save()

    def finish(self):
        # 自生漂移率（SDR）：所有過閘門的記憶裡，來源不是使用者真實語音的比例
        passing, drift, wrong_owner = 0, 0, 0
        rows = []
        for (e, s), pv in list(self.brain.prov.items()):
            ans, th, pm, ok = self.brain.recall(e, s, bump=False)
            if not ok:
                continue
            passing += 1
            bad = pv["src"] != "owner"
            drift += bad
            rows.append({"anchor": f"{e}|{s}", "src": pv["src"], "answer": ans, "pmin": round(pm, 3)})
        self.sdr = {"passing": passing, "not_from_owner": drift, "SDR": round(drift / max(1, passing), 4), "rows": rows}

    def save(self, partial=False):
        out = {"mode": self.a.mode, "config": vars(self.a), "partial": partial, "hour": self.hour, "log": self.log,
               "agenda_left": self.agenda, "timing": {"gen_tok_s": self.tok_s(), "ttft_s": self.ttft()}}
        if hasattr(self.brain, "calls"):          # E41：真記憶每次寫入、查熟悉感實際花的秒數
            out["mem_calls"] = {k: [round(x, 3) for x in v] for k, v in self.brain.calls.items()}
        if not partial:
            out["sdr"] = self.sdr
            out["wall_s"] = round(self.wall, 1)
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        json.dump(out, open(os.path.join(HERE, "results", f"run_{self.a.mode}_{self.a.days}d_s{self.a.seed}{self.a.out_tag}.json"),
                            "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--tag", default="21d")
    p.add_argument("--D", type=int, default=8192)
    p.add_argument("--pmin", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sleep_every", type=int, default=50)
    p.add_argument("--idle_tokens", type=int, default=48)
    p.add_argument("--b_tokens", type=int, default=64)
    p.add_argument("--hours", type=float, default=0, help="只跑前幾個小時（測試用）")
    p.add_argument("--out_tag", default="")
    p.add_argument("--thinker", choices=["local", "llm"], default="local")
    p.add_argument("--event_log", action="store_true", help="E33/U15：網路只當熟悉感閘門，內容從事件紀錄取最新值")
    p.add_argument("--memory", choices=["local", "llm"], default="local", help="E41：llm = AI 電腦上的真記憶")
    p.add_argument("--find", default="", help="E41：embed = 沒有完全相同的標籤時用語意找最近的標籤並反問")
    p.add_argument("--find_tau", type=float, default=0.6)
    p.add_argument("--write_behind", action="store_true", help="H26：網路寫入排到空檔，不讓使用者等")
    p.add_argument("--wb_gap", type=float, default=10.0, help="H26：使用者停下來幾秒後才開始寫")
    p.add_argument("--cache", default="zh_result_E_log.json", help="E44：聆聽快取（llm_fastmem/results 裡的 zh_v3 結果檔）")
    p.add_argument("--read", action="store_true", help="E44/H28：問題用搜原句再讀回答（需要 --memory llm）")
    p.add_argument("--read_phon", action="store_true", help="E44/H31b：說不知道時拼音相同的紀錄再讀一次")
    p.add_argument("--embed_idle", action="store_true", help="E45/H34：紀錄的向量在空檔寫入時算好，不在使用者問的時候算")
    p.add_argument("--index", action="store_true", help="E46/H35：網路索引（字詞第 0 層向量＋拼音 → 紀錄編號）負責找紀錄")
    p.add_argument("--no_gate", action="store_true", help="E47：回答時不對挑中的紀錄查熟悉感")
    p.add_argument("--clean", action="store_true", help="E48/H40：聆聽寫出的問句與註解不寫進記憶")
    p.add_argument("--index_split", action="store_true", help="E48/H38a：網路索引的意義與拼音單元分開配額")
    p.add_argument("--ortho", action="store_true", help="E49/H41：網路索引前 512 筆紀錄用彼此正交的編碼（排序不再受隨機種子影響）")
    Sim(p.parse_args()).run()
