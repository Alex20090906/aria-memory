"""H35 (E46): the network does the finding. A hippocampal-style index in fast weights maps what was heard to the
number of a logged row; the row itself (fact + original sentence) stays in the event log.

Key of a text: a sparse activation over 65536 units from layer-0 word vectors and sound only.
  meaning bank (units 0..32767): each LLM server token -> its layer-0 vector (LLM server's own token embeddings), normalised,
                                 times a fixed random projection -> top 4 units;
  sound bank (32768..65535):     each toneless pinyin syllable and syllable pair (confusable sounds merged) -> 2 hashed units.
Units are scaled by 1/sqrt(n_i + 10) (n_i = rows that used unit i), the 128 strongest are kept, the key is normalised.
Value: a fixed random +-1/sqrt(512) code per row. Write: W += z (c - 0.2 W^T z)^T (delta rule: a topic heard again
leans to its latest rows). Read: y = W^T z(q); rows ranked by c(u) . y.

Offline (p8/index_eval.py, E43 logs): top-5 recall 35/38 and 35/38 (text embedding 35, 34); end to end with the reader
32 and 39 of 51 right (text embedding 32, 38), the same wrong answers and never-told answers.
Word vectors: rows of LLM server's token_embd for the tokens these logs use (p8/llm_tok, copied from the AI PC).
"""
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.path.join(HERE, "..", "p8", "llm_tok")
D, H, DOUT, K, KT, KS, GAMMA = 65536, 32768, 512, 128, 4, 2, 0.2
CONF = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]


def _h32(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


class NetIndex:
    def __init__(self, seed=1):
        from transformers import Qwen2TokenizerFast
        self.tok = Qwen2TokenizerFast(vocab_file=os.path.join(TOK_DIR, "vocab.json"),
                                      merges_file=os.path.join(TOK_DIR, "merges.txt"))
        ids = np.load(os.path.join(TOK_DIR, "ids.npy"))
        rows = np.load(os.path.join(TOK_DIR, "emb_rows.npy")).astype(np.float32)
        self.E = {int(i): r for i, r in zip(ids, rows)}
        rng = np.random.default_rng(seed)
        s = np.load(os.path.join(TOK_DIR, "emb_sample.npy")).astype(np.float32)
        s = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-5)
        self.mu, self.sd = s.mean(0), s.std(0) + 1e-3
        self.P = (rng.standard_normal((rows.shape[1], H)) / np.sqrt(rows.shape[1])).astype(np.float32)
        self.W = np.zeros((D, DOUT), np.float32)
        self.n_use = np.zeros(D, np.float32)
        self.C = []                      # row codes, in row order
        self._crng = np.random.default_rng(7)
        self.ortho = None          # H41: exactly orthogonal codes for the first DOUT rows (set by use_ortho)
        self._ucache = {}
        self.missing_tokens = 0

    def _token_units(self, t):
        u = self._ucache.get(t)
        if u is None:
            x = self.E.get(t)
            if x is None:                # a token without a copied word vector: sound units only
                self.missing_tokens += 1
                u = np.zeros(0, np.int64)
            else:
                x = ((x - x.mean()) / (x.std() + 1e-5) - self.mu) / self.sd
                u = np.argpartition(-(x @ self.P), KT)[:KT]
            self._ucache[t] = u
        return u

    def _raw(self, text):
        from pypinyin import lazy_pinyin, Style
        cnt = {}
        for t in self.tok(text).input_ids:
            piece = self.tok.decode([t]).strip()
            if not piece or not re.search(r"[\w一-鿿]", piece):
                continue
            for u in self._token_units(t):
                cnt[int(u)] = cnt.get(int(u), 0) + 1
        syl = []
        for ch in text:
            if "一" <= ch <= "鿿":
                s = lazy_pinyin(ch, style=Style.NORMAL)[0]
                for a, b in CONF:
                    s = s.replace(a, b)
                syl.append(s)
        for ft in [f"u:{s}" for s in syl] + [f"b:{a}_{b}" for a, b in zip(syl, syl[1:])]:
            for j in range(KS):
                u = H + _h32(f"{ft}#{j}") % H
                cnt[u] = cnt.get(u, 0) + 1
        return cnt

    split = False          # H38a (E48): K/2 meaning units + K/2 sound units chosen separately (offline +1 of 76)

    @staticmethod
    def _topk(units, w, k):
        if len(units) > k:
            keep = np.argpartition(-w, k)[:k]
            units, w = units[keep], w[keep]
        return units, w / (np.linalg.norm(w) + 1e-9)

    def _sparse(self, cnt):
        units = np.array(list(cnt), np.int64)
        w = np.array([cnt[u] for u in units], np.float32) / np.sqrt(self.n_use[units] + 10.0)
        if not self.split:
            return self._topk(units, w, K)
        m = units < H
        u1, w1 = self._topk(units[m], w[m], K // 2)
        u2, w2 = self._topk(units[~m], w[~m], K // 2)
        return np.concatenate([u1, u2]), np.concatenate([w1, w2]) * np.float32(np.sqrt(0.5))

    def write(self, text):
        """add the next row (its number is len(C) before the call)"""
        c = (self._crng.choice(np.array([-1.0, 1.0], np.float32), DOUT) / np.sqrt(DOUT)).astype(np.float32)
        if self.ortho is not None and len(self.C) < DOUT:
            c = self.ortho[len(self.C)]     # H41: a row's score is then its key overlap with the question, no code noise
        self.C.append(c)
        cnt = self._raw(text)
        self.n_use[list(cnt)] += 1
        u, w = self._sparse(cnt)
        old = w @ self.W[u]
        self.W[u] += np.outer(w, c - GAMMA * old)

    # ---------------------------------------------------------------- persistence ("always alive" across restarts)
    def save(self, path):
        """atomic save of everything the index learned: W, the unit use counts, the row codes and the code generator's
        state (so rows written after a restart get the same codes as without the restart)"""
        tmp = path + ".tmp.npz"
        np.savez(tmp, W=self.W, n_use=self.n_use, C=np.stack(self.C) if self.C else np.zeros((0, DOUT), np.float32),
                 rng=np.array(json.dumps(self._crng.bit_generator.state)))
        os.replace(tmp, path)

    def load(self, path):
        z = np.load(path, allow_pickle=False)
        self.W, self.n_use = z["W"].copy(), z["n_use"].copy()
        self.C = list(z["C"])
        self._crng.bit_generator.state = json.loads(str(z["rng"]))

    def use_ortho(self, seed=7):
        """H41: rows 0..DOUT-1 get the rows of a fixed random orthogonal matrix as codes. Offline (E43 logs, 3 seeds):
        top-5 recall 35/38 and 34/38 for every seed (random codes: 33-35 depending on the seed)."""
        q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((DOUT, DOUT)))
        self.ortho = q.astype(np.float32)

    def query(self, text, k=5):
        if not self.C:
            return []
        u, w = self._sparse(self._raw(text))
        y = w @ self.W[u]
        s = np.stack(self.C) @ y
        return np.argsort(-s)[:k].tolist()
