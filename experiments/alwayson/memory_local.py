"""本機的思考模型替身＋快權重記憶（AI 電腦空出來之前用）。

思考模型：Qwen2.5-1.5B-Instruct（本機 RTX 4060）。記憶的做法與 語言模型伺服器 版相同（memory/metamemory_qwen.py、
llm_fastmem/sleep.py），只是搬到中文錨點：
  鍵   = 第 Ls 層「東西」與「屬性」兩段的雙線性綁定 → 取 k 個單元；答案第 t 個字的鍵 = 錨點單元 + t 位移（不看前一個字）
  值   = 加在輸出層的輸入（final norm 之後）；寫入 = 只用前向計算的誤差驅動規則
         v = dl·(1-p_y)·(u_y - E_p[u]) / |u_y - E_p[u]|²
  θ    = 錨點單元的熟悉感；每次寫入衰減，寫入時設回 1
  τ    = 每個單元的時間常數（快 100 次寫入、睡眠升格後慢 500）；權重每次寫入乘 exp(-1/τ)
  c    = 被成功想起的次數
回答時的閘門：θ 平均 ≥ 0.5 而且整句最弱字機率 ≥ PMIN（H14b/H21：只在回答時用，不在睡眠重播時用）。
睡眠：選最近 64 個錨點中 θ 高且被想起過的重播；模型自己補完（真實性檢查不過就不鞏固）；
其餘未升格、也不屬於最近清醒窗口的單元 ×0.1；通過的重新補寫並升格 τ。
"""
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TAU_FAST, TAU_SLOW, DOWNSCALE = 100.0, 500.0, 0.1
TH_TAU_FAST, TH_TAU_SLOW = 500.0, 5000.0
BAD_VALUE = re.compile(r"(\[|\]|\||->|不知道|不記得|沒有|NONE|無法|抱歉)")


class Brain:
    """思考模型替身與它權重裡的快權重記憶。chat() 時記憶關閉；recall() 時注入。"""

    def __init__(self, D=8192, k=16, Ls=0, dl=24.0, pmin=0.7, gate=0.5, seed=0, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device).eval()
        self.dev = device
        H = self.model.config.hidden_size
        self.H, self.D, self.k, self.Ls, self.dl, self.pmin, self.gate = H, D, k, Ls, dl, pmin, gate
        g = torch.Generator().manual_seed(seed)
        self.P = (torch.randn(H, D, generator=g) / H ** 0.5).to(device)
        self.B = torch.zeros(D, H, device=device)
        self.theta = torch.zeros(D, device=device)
        self.tau = torch.full((D,), TAU_FAST, device=device)
        self.c = torch.zeros(D, device=device)
        self.U = self.model.lm_head.weight.float()
        self.st = {"units": None, "off": True}
        self.stats = None
        self.ring = []                       # 最近 64 個寫入的錨點（只存名字）
        self.writes = 0
        self.prov = {}                       # (e, s) -> 最後一次寫入的來源（評分用，不給系統看）
        self.elog, self.log_mode = [], False   # U15 事件紀錄
        self.timing = {"gen_tok_s": [], "prefill_s": []}
        self.model.model.norm.register_forward_hook(self._hook)

    # ------------------------------------------------------------------ internals
    def _hook(self, mod, inp, out):
        if self.st["off"] or self.st["units"] is None:
            return out
        T, p0 = out.shape[1], self.st.get("pos0", 0)
        add = torch.zeros(T, self.H, device=out.device)
        for t, u in self.st["units"].items():
            if 0 <= t - p0 < T:
                add[t - p0] = self.B[u].sum(0) / self.k ** 0.5
        return out + add.to(out.dtype)[None]

    def ids(self, s):
        return self.tok(s, return_tensors="pt").input_ids.to(self.dev)

    def _pieces(self, x):
        return [self.tok.decode([int(i)]) for i in x[0]]

    def _norm(self, v):
        return nn.functional.layer_norm(v.float(), (self.H,))

    @torch.no_grad()
    def _hs(self, x):
        self.st["off"] = True
        if self.Ls == 0:                    # 第 0 層 = 詞向量本身，不受前文影響，也不必跑整個模型
            return [self.model.model.embed_tokens(x)]
        return self.model(x, output_hidden_states=True).hidden_states

    def _spans(self, x):
        p = self._pieces(x)
        a_end = max(i for i, s in enumerate(p) if "->" in s)
        lb = max(i for i in range(a_end) if "[" in p[i])
        bar = max(i for i in range(lb, a_end) if "|" in p[i])
        rb = max(i for i in range(bar, a_end + 1) if "]" in p[i])
        subj = list(range(lb + 1, bar)) or [lb]
        obj = list(range(bar + 1, rb)) or [bar]
        return a_end, subj, obj

    def calibrate(self, pairs):
        """鍵的標準化統計（每一維的平均與標準差），用一批真實的 [東西|屬性] 計算一次。"""
        acc = {"s": [], "o": []}
        for e, s in pairs:
            x = self.ids(f"[{e}|{s}] ->")
            hs = self._hs(x)
            _, subj, obj = self._spans(x)
            acc["s"].append(self._norm(hs[self.Ls][0, subj]).mean(0)[None])
            acc["o"].append(self._norm(hs[self.Ls][0, obj]).mean(0)[None])
        self.stats = {k_: (torch.cat(v).mean(0), torch.cat(v).std(0) + 1e-3) for k_, v in acc.items()}

    def _anchor(self, x):
        hs = self._hs(x)
        a_end, subj, obj = self._spans(x)
        xs = (self._norm(hs[self.Ls][0, subj]).mean(0) - self.stats["s"][0]) / self.stats["s"][1]
        xo = (self._norm(hs[self.Ls][0, obj]).mean(0) - self.stats["o"][0]) / self.stats["o"][1]
        sc = (xs @ self.P) * torch.roll(xo @ self.P, 1)
        return sc.topk(self.k).indices, a_end

    def _units(self, x):
        anc, a_end = self._anchor(x)
        out = {a_end: anc}
        for t in range(a_end + 1, x.shape[1]):
            out[t] = (anc + (t - a_end) * 1000003) % self.D
        return out, anc, a_end

    def anchor_units(self, e, s):
        return self._anchor(self.ids(f"[{e}|{s}] ->"))[0]

    def familiarity(self, e, s):
        return float(self.theta[self.anchor_units(e, s)].mean())

    # ------------------------------------------------------------------ memory operations
    @torch.no_grad()
    def _decay(self):
        self.B.mul_(torch.exp(-1.0 / self.tau)[:, None])
        th_tau = torch.where(self.tau >= TAU_SLOW, torch.tensor(TH_TAU_SLOW, device=self.dev),
                             torch.tensor(TH_TAU_FAST, device=self.dev))
        self.theta.mul_(torch.exp(-1.0 / th_tau))

    @torch.no_grad()
    def wipe(self, e, s, n_steps=24):
        """更新前抹除同一個 [東西|屬性] 的舊答案（H10）：清掉答案位置的列，不動錨點本身。"""
        anc = self.anchor_units(e, s)
        for t in range(1, n_steps + 1):
            self.B[(anc + t * 1000003) % self.D] = 0

    @torch.no_grad()
    def write(self, e, s, v, src="owner", decay=True):
        if decay:
            self._decay()
        if self.familiarity(e, s) >= self.gate:
            self.wipe(e, s)
        x = self.ids(f"[{e}|{s}] -> {v}.")
        u, anc, _ = self._units(x)
        self.theta[anc] = 1.0
        self.st["units"], self.st["off"] = u, False
        pr = self.model(x).logits[0].float()[:-1].softmax(-1)
        self.st["off"] = True
        for t, units in u.items():
            if t + 1 >= x.shape[1]:
                continue
            y = x[0, t + 1]
            du = self.U[y] - pr[t] @ self.U
            self.B[units] += (self.dl * (1 - pr[t, y]) * du / (du @ du + 1e-6)) / self.k ** 0.5
        self.ring = (self.ring + [(e, s)])[-64:]
        self.prov[(e, s)] = {"src": src, "value": v, "at_write": self.writes}
        self.elog.append(((e, s), v))                 # U15：只增不改的事件紀錄（只有 log_mode 會讀它）
        self.writes += 1

    @torch.no_grad()
    def recall(self, e, s, question="", n_new=16, bump=True, gated=True):
        """回想 [e|s] -> ...。回傳 (答案, θ, 最弱字機率, 是否過閘)。gated=False 時一律讀出（睡眠重播用）。"""
        anc = self.anchor_units(e, s)
        th = float(self.theta[anc].mean())
        if gated and th < self.gate:
            return "", th, 0.0, False
        if self.log_mode:
            # U15（llm_fastmem/sleep.py 的 SLEEP_LOG）：網路只當熟悉感閘門；過了閘門，內容從事件紀錄依標籤取最新值。
            # 標籤就是 (e, s) 這兩個字串，不做模糊比對，所以問法和寫法不一樣時會翻不到（P8）。
            if th < self.gate:
                return "", th, 0.0, False
            val = next((v for k, v in reversed(self.elog) if k == (e, s)), "")
            if bump and val:
                self.c[anc] += 1
            return val, th, 1.0 if val else 0.0, bool(val)
        prompt = (question + "\n" if question else "") + f"[{e}|{s}] ->"
        x = self.ids(prompt)
        n0, ps = x.shape[1], []
        anc_q, a_end = self._anchor(x)                 # 錨點只算一次；答案第 t 個字的鍵 = 錨點 + t 位移
        # 記憶只加在輸出層的輸入，不改前面各層，所以可以用 KV 快取一次只算一個新字（結果與整段重算相同）
        past, step_in, pos0 = None, x, 0
        for _ in range(n_new):
            t_last = x.shape[1] - 1
            self.st["units"] = {t_last: (anc_q + (t_last - a_end) * 1000003) % self.D}
            self.st["pos0"], self.st["off"] = pos0, False
            o = self.model(step_in, past_key_values=past, use_cache=True)
            self.st["off"], self.st["pos0"] = True, 0
            past, lg = o.past_key_values, o.logits[0, -1].float()
            pos0 = x.shape[1]
            y = int(lg.argmax())
            piece = self.tok.decode([y])
            if "." in piece or "\n" in piece or "。" in piece or "[" in piece or "->" in piece:
                break
            ps.append(float(lg.softmax(-1)[y]))
            x = torch.cat([x, torch.tensor([[y]], device=self.dev)], 1)
            step_in = x[:, -1:]
        ans = self.tok.decode(x[0, n0:]).strip()
        pm = min(ps) if ps else 0.0
        ok = bool(ans) and (not gated or pm >= self.pmin)
        if bump and ok:
            self.c[anc] += 1
        return ans, th, pm, ok

    @torch.no_grad()
    def sleep(self, last_window):
        """睡眠鞏固（llm_fastmem/sleep.py 的同一套規則）。last_window = 最近一個清醒窗口寫入的錨點。"""
        replay = []
        for e, s in dict.fromkeys(self.ring):
            anc = self.anchor_units(e, s)
            if float(self.theta[anc].mean()) >= 0.5 and float((self.c[anc] >= 1).float().mean()) >= 0.5:
                replay.append((e, s, anc))
        cons, rewrites, rejected = set(), [], 0
        for e, s, anc in replay:
            val, _, _, _ = self.recall(e, s, bump=False, gated=False)
            if not val or len(val) > 30 or BAD_VALUE.search(val):      # 真實性檢查：補出來不像答案就不鞏固
                rejected += 1
                continue
            rewrites.append((e, s, val))
            x = self.ids(f"[{e}|{s}] -> {val}.")
            u, _, _ = self._units(x)
            for units in u.values():
                cons.update(units.tolist())
        mask = self.tau < TAU_SLOW
        if cons:
            mask[list(cons)] = False
        for e, s in dict.fromkeys(last_window):               # 最近清醒窗口還沒被問過，不壓低也不升格
            val, _, _, _ = self.recall(e, s, bump=False, gated=False)
            x = self.ids(f"[{e}|{s}] -> {val}.") if val else self.ids(f"[{e}|{s}] ->")
            u, _, _ = self._units(x)
            for units in u.values():
                mask[units] = False
        self.B[mask] *= DOWNSCALE
        for e, s, val in rewrites:
            src = self.prov.get((e, s), {}).get("src", "owner")
            self.write(e, s, val, src=src, decay=False)
        if cons:
            self.tau[list(cons)] = TAU_SLOW
        return {"replayed": len(rewrites), "candidates": len(replay), "rejected": rejected, "consolidated_units": len(cons),
                "downscaled_units": int(mask.sum())}

    def participation_ratio(self):
        w = self.B.norm(dim=1)
        s = float(w.sum())
        return (s * s / float((w * w).sum())) if s > 0 else 0.0

    # ------------------------------------------------------------------ thinking (memory off)
    @torch.no_grad()
    def chat(self, system, user, max_new=96, shots=(), temperature=0.0):
        msgs = [{"role": "system", "content": system}]
        for a, b in shots:
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": user})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = self.ids(text)
        self.st["off"] = True
        t0 = time.time()
        self.model(x[:, :])                                  # 量首字延遲（prefill）
        torch.cuda.synchronize()
        t1 = time.time()
        kw = {"do_sample": temperature > 0, "max_new_tokens": max_new, "pad_token_id": self.tok.eos_token_id}
        if temperature > 0:
            kw.update(temperature=temperature, top_p=0.95)
        out = self.model.generate(x, **kw)
        torch.cuda.synchronize()
        t2 = time.time()
        new = out[0, x.shape[1]:]
        n = int(len(new))
        self.timing["prefill_s"].append(t1 - t0)
        if n:
            self.timing["gen_tok_s"].append((t2 - t1 - (t1 - t0)) / n if (t2 - t1) > (t1 - t0) else (t2 - t1) / n)
        return self.tok.decode(new, skip_special_tokens=True).strip(), n, len(x[0])

    def n_tokens(self, s):
        return len(self.tok(s).input_ids)
