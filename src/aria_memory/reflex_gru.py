"""The research's reflex model (model 1) in numpy, for parity only: it is NOT wired into the gate.

Model1 (research experiments/model1/train_gru3.py, weights w6r_qwen3emb_fam_s0.pt): Linear(1029 -> 128) + GELU,
GRU(128), heads: 5 actions (WAIT, ATTEND, REPLY, THINK, INTERRUPT) and 1 memory-encode gate. 231,686 parameters.
One tick per second; the hidden state is never reset.
Input per tick (1029): the Qwen3-Embedding-0.6B vector (1024, normalised) of the tick's event text
("[MIC] owner: …", "[MIC] someone: …", "[CAM] …"; zeros when nothing happened), then
[has_event, familiarity, thinker_busy, busy_for, fast_familiarity].

Why it is not wired: it needs a Qwen3-Embedding-0.6B vector per event (not available on CPU in this product), and in
E49 its output only supplied the camera novelty +1 energy. Owner routing used the voiceprint, not this model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ACTIONS = ["WAIT", "ATTEND", "REPLY", "THINK", "INTERRUPT"]
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ReflexGRU:
    def __init__(self, weights: dict):
        g = {k: np.asarray(v, np.float32) for k, v in weights.items()}
        self.Wi, self.bi = g["inp.0.weight"], g["inp.0.bias"]
        self.Wih, self.Whh = g["rnn.weight_ih_l0"], g["rnn.weight_hh_l0"]
        self.bih, self.bhh = g["rnn.bias_ih_l0"], g["rnn.bias_hh_l0"]
        self.Wo, self.bo = g["out.weight"], g["out.bias"]
        self.We, self.be = g["enc.weight"], g["enc.bias"]
        self.hidden = self.Whh.shape[1]
        self.d_in = self.Wi.shape[1]

    @classmethod
    def load(cls, path: str | Path) -> "ReflexGRU":
        z = np.load(path)
        return cls({k: z[k] for k in z.files})

    def n_params(self) -> int:
        return sum(a.size for a in (self.Wi, self.bi, self.Wih, self.Whh, self.bih, self.bhh, self.Wo, self.bo,
                                    self.We, self.be))

    def step(self, x: np.ndarray, h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """One tick: (6 outputs = 5 action logits + encode logit, new hidden state)."""
        h = np.zeros(self.hidden, np.float32) if h is None else h
        a = _gelu(self.Wi @ np.asarray(x, np.float32) + self.bi)
        gi = self.Wih @ a + self.bih
        gh = self.Whh @ h + self.bhh
        n_ = self.hidden
        r = _sig(gi[:n_] + gh[:n_])
        z = _sig(gi[n_:2 * n_] + gh[n_:2 * n_])
        n = np.tanh(gi[2 * n_:] + r * gh[2 * n_:])
        h2 = ((1.0 - z) * n + z * h).astype(np.float32)
        out = np.concatenate([self.Wo @ h2 + self.bo, self.We @ h2 + self.be]).astype(np.float32)
        return out, h2

    def decide(self, x: np.ndarray, h: np.ndarray | None = None) -> tuple[str, bool, np.ndarray]:
        out, h2 = self.step(x, h)
        return ACTIONS[int(np.argmax(out[:5]))], bool(out[5] > 0), h2
