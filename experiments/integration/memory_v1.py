"""Memory model v1 as a reusable component (see experiments/memory/inllm.py for the experiment).

The memory lives in weights only (no database of texts):
  w     : sparse fast-weight adapter B on the residual stream of a local Qwen2.5-1.5B,
          trained online on each experienced event, decays with tau_w (per event)
  theta : slow familiarity traces on adapted (background-subtracted) sparse codes of the
          model's own hidden state, decay with tau_theta (per event)
Recall is pattern completion: the model continues a cue with the adapter on.
"""
import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class MemoryModel:
    def __init__(self, layer=14, D=16384, k=16, lr=3e-2, steps=10, tau_w=40.0, tau_theta=2000.0,
                 Df=32768, kf=20, adapt_floor=0.005, seed=0, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        H = self.H = self.model.config.hidden_size
        g = torch.Generator().manual_seed(seed)
        self.P = (torch.randn(H, D, generator=g) / H ** 0.5).to(device)
        self.Pf = (torch.randn(H, Df, generator=g) / H ** 0.5).to(device)
        self.B = torch.zeros(D, H, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.B], lr=lr)
        self.k, self.kf, self.steps = k, kf, steps
        self.dw, self.dth = math.exp(-1 / tau_w), math.exp(-1 / tau_theta)
        self.codes, self.strength = [], []
        self.mu, self.var, self.n, self.adapt_floor = None, None, 0, adapt_floor
        self.on, self.cap = True, {}
        # token-level sensory adaptation for the fast-weight code (same idea as for theta)
        self.tmu, self.tvar = None, None
        self.model.model.layers[layer].register_forward_hook(self._hook)
        self.device = device

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.cap["h"] = h
        if not self.on:
            return out
        hn = nn.functional.layer_norm(h.float(), (self.H,))
        if self.tmu is not None:
            hn = (hn - self.tmu) / (self.tvar.sqrt() + 1e-3)
        a = hn @ self.P
        z = torch.zeros_like(a)
        z.scatter_(-1, a.topk(self.k, dim=-1).indices, 1 / self.k ** 0.5)
        h2 = h + (z @ self.B).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    def _ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def _code(self, text, learn=False):
        on, self.on = self.on, False
        with torch.no_grad():
            self.model(self._ids(text))
        self.on = on
        toks = nn.functional.layer_norm(self.cap["h"][0, 1:].float(), (self.H,))
        h = toks.mean(0)
        if learn:
            self.n += 1
            a = max(1 / self.n, self.adapt_floor)
            tm, tv = toks.mean(0), toks.var(0, unbiased=False)
            if self.tmu is None:
                self.tmu, self.tvar = tm.clone(), tv + 1e-2
            else:
                self.tvar = (1 - a) * self.tvar + a * (tv + (tm - self.tmu) ** 2)
                self.tmu = (1 - a) * self.tmu + a * tm
            if self.mu is None:
                self.mu, self.var = h.clone(), torch.ones_like(h)
            else:
                a = max(1 / self.n, self.adapt_floor)
                self.mu = (1 - a) * self.mu + a * h
                self.var = (1 - a) * self.var + a * (h - self.mu) ** 2
        if self.mu is not None:
            h = (h - self.mu) / (self.var.sqrt() + 1e-3)
        s = h @ self.Pf
        c = torch.zeros_like(s)
        c[s.topk(self.kf).indices] = 1 / self.kf ** 0.5
        return c

    GIST_SYS = ("Rewrite what was heard or seen as ONE short fact in plain words, using the pattern "
                "\"<owner>'s <thing> is <where/what>.\" when it is about where something is. "
                "A question is not a fact: write \"<who> asked <what>.\" "
                "Otherwise one short sentence \"<who> <did what>.\". Output only the fact.")
    GIST_SHOTS = [("Mia said: just so you know, I dropped my car keys into the green bowl.",
                   "Mia's car keys is in the green bowl."),
                  ("Tom said: FYI the hammer went back to the shed.", "Tom's hammer is in the shed."),
                  ("Ken said: where did I leave my phone?", "Ken asked where his phone is."),
                  ("Sara said: I can't find my bag, any idea where it went?", "Sara asked where her bag is."),
                  ("I saw: a dog runs across the garden.", "A dog ran across the garden.")]

    @torch.no_grad()
    def gist(self, text):
        """Cortex-like interpretation before storage: the same model, memory off, restates the event
        in a canonical form. Memory then stores the meaning, not the exact wording."""
        msgs = [{"role": "system", "content": self.GIST_SYS}]
        for a, b in self.GIST_SHOTS:
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": text})
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        x = ids["input_ids"].to(self.device)
        on, self.on = self.on, False
        out = self.model.generate(x, max_new_tokens=30, do_sample=False, pad_token_id=self.tok.eos_token_id)
        self.on = on
        return self.tok.decode(out[0, x.shape[1]:], skip_special_tokens=True).strip().splitlines()[0]

    def experience(self, text):
        """Live through one event: write details into w, leave a familiarity trace in theta."""
        code = self._code(text, learn=True)
        x = self._ids(text)
        self.on = True
        for _ in range(self.steps):
            lg = self.model(x).logits[0, :-1].float()
            loss = nn.functional.cross_entropy(lg, x[0, 1:])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        with torch.no_grad():
            self.B.mul_(self.dw)
        self.strength = [s * self.dth for s in self.strength]
        self.codes.append(code)
        self.strength.append(1.0)

    def warm(self, text):
        """Perceive without storing: only updates sensory adaptation statistics."""
        self._code(text, learn=True)

    def familiarity(self, text):
        if not self.codes:
            return 0.0
        c = self._code(text)
        C = torch.stack(self.codes)
        S = torch.tensor(self.strength, device=c.device)
        return float(((C @ c) ** 2 * S).sum())

    @torch.no_grad()
    def _complete(self, cue, n):
        x = self._ids(cue)
        out = self.model.generate(x, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)

    def recall(self, cue, n=12):
        """Pattern completion. Also says whether the memory changed what the model would say
        anyway (prior), and how familiar the cue and the recalled sentence feel."""
        self.on = True
        mem = self._complete(cue, n)
        self.on = False
        prior = self._complete(cue, n)
        self.on = True
        first = lambda s: s.split(".")[0].strip()
        full = cue + " " + first(mem) + "."
        return {"cue": cue, "recalled": first(mem), "differs_from_guess": first(mem) != first(prior),
                "cue_familiarity": round(self.familiarity(cue), 3),
                "recalled_familiarity": round(self.familiarity(full), 3)}
