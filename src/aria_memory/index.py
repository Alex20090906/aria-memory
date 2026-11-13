"""The memory index: a hippocampal-style index in fast weights that maps what was heard to record ids (E49, H35/H41).

Key of a text: a sparse activation over 65536 units, from the thinker model's layer-0 word vectors and from sound only.
  meaning bank (units 0..32767)   each model token -> its 4 precomputed meaning units (vocab.Vocab)
  sound bank (32768..65535)       each toneless pinyin syllable and syllable pair (zh/z, ch/c, sh/s, ang/an, eng/en,
                                  ing/in merged) -> 2 hashed units
Units are weighted by 1/sqrt(n_i + 10) (n_i = records that used unit i), the 128 strongest kept, the key normalised.

Value: a code per record position. Block-orthogonal (H41b): every block of 512 consecutive positions gets the rows of
its own fixed random orthogonal matrix, so within a block a record's score is its key overlap with the question and
nothing else. Block 0 is E49's matrix (seed 7); the score does not depend on which orthogonal basis is used.

Write (delta rule): W += z (c - 0.2 W^T z)^T, so a topic heard again leans to its latest records.
Read: y = W^T z(question); records ranked by c . y.

Memory: W is 65536 x 512 float32 (128 MB). Records themselves are not stored here; the caller keeps them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .records import key_text

D, H = 65536, 32768
K, KS, GAMMA = 128, 2, 0.2
CONF = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]
_SYL: dict[str, str] = {}


def _h32(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def syllable(ch: str) -> str:
    s = _SYL.get(ch)
    if s is None:
        from pypinyin import Style, lazy_pinyin

        s = lazy_pinyin(ch, style=Style.NORMAL)[0]
        for a, b in CONF:
            s = s.replace(a, b)
        _SYL[ch] = s
    return s


def is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


class NetIndex:
    def __init__(self, vocab, dim: int = 512, block_orthogonal: bool = True, seed: int = 7):
        self.vocab = vocab
        self.dim = dim
        self.block_orthogonal = block_orthogonal
        self.seed = seed
        self.W = np.zeros((D, dim), np.float32)
        self.n_use = np.zeros(D, np.float32)
        self.ids: list = []                  # position -> record id
        self.pos: dict = {}                  # record id -> position
        self._blocks: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids)

    # ---------------------------------------------------------------- codes
    def _block(self, b: int) -> np.ndarray:
        q = self._blocks.get(b)
        if q is None:
            rng = np.random.default_rng(self.seed if b == 0 else (self.seed, b))
            q, _ = np.linalg.qr(rng.standard_normal((self.dim, self.dim)))
            q = q.astype(np.float32)
            self._blocks[b] = q
        return q

    def code(self, p: int) -> np.ndarray:
        if self.block_orthogonal or p < self.dim:
            return self._block(p // self.dim)[p % self.dim]
        rng = np.random.default_rng((self.seed, 1_000_000 + p))       # E49 beyond 512 rows: random +-1 codes
        return (rng.choice(np.array([-1.0, 1.0], np.float32), self.dim) / np.sqrt(self.dim)).astype(np.float32)

    def codes(self, positions) -> np.ndarray:
        positions = list(positions)
        if not positions:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self.code(p) for p in positions])

    # ---------------------------------------------------------------- keys
    def raw(self, text: str) -> dict[int, float]:
        """unit -> count, before the frequency weighting"""
        cnt: dict[int, float] = {}
        for t in self.vocab.token_ids(text):
            for u in self.vocab.units(t):
                cnt[int(u)] = cnt.get(int(u), 0) + 1
        syl = [syllable(ch) for ch in text or "" if is_cjk(ch)]
        for ft in [f"u:{s}" for s in syl] + [f"b:{a}_{b}" for a, b in zip(syl, syl[1:])]:
            for j in range(KS):
                u = H + _h32(f"{ft}#{j}") % H
                cnt[u] = cnt.get(u, 0) + 1
        return cnt

    def sparse(self, cnt: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        units = np.array(list(cnt), np.int64)
        if len(units) == 0:
            return units, np.zeros(0, np.float32)
        w = np.array([cnt[u] for u in units], np.float32) / np.sqrt(self.n_use[units] + 10.0)
        if len(units) > K:
            keep = np.argpartition(-w, K)[:K]
            units, w = units[keep], w[keep]
        return units, w / (np.linalg.norm(w) + 1e-9)

    def readout(self, text: str) -> np.ndarray:
        u, w = self.sparse(self.raw(text))
        return w @ self.W[u] if len(u) else np.zeros(self.dim, np.float32)

    def units_of(self, text: str) -> set[int]:
        """the key's units (used by the gate's overlap fuse)"""
        return set(self.sparse(self.raw(text))[0].tolist())

    # ---------------------------------------------------------------- write
    def add(self, rec_id, record: dict) -> None:
        """Add a record (its key text is thing, aspect, value and heard sentence, as in E49)."""
        self.add_text(rec_id, key_text(record))

    def add_text(self, rec_id, text: str) -> None:
        if rec_id in self.pos:
            raise ValueError(f"record {rec_id!r} is already in the index")
        p = len(self.ids)
        c = self.code(p)
        cnt = self.raw(text)
        self.n_use[list(cnt)] += 1
        u, w = self.sparse(cnt)
        if len(u):
            old = w @ self.W[u]
            self.W[u] += np.outer(w, c - GAMMA * old)
        self.ids.append(rec_id)
        self.pos[rec_id] = p

    # ---------------------------------------------------------------- read
    def query(self, text: str, k: int = 5, exclude=()) -> list[tuple]:
        """Top-k (record id, score), best first. `exclude`: ids not to return (e.g. superseded records)."""
        if not self.ids:
            return []
        ex = set(exclude)
        positions = [p for p, i in enumerate(self.ids) if i not in ex]
        if not positions:
            return []
        s = self.codes(positions) @ self.readout(text)
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[positions[j]], float(s[j])) for j in top]

    def scores(self, text: str, rec_ids) -> list[tuple]:
        """(record id, score) for the given ids, best first (ids not in the index are skipped)."""
        rec_ids = [i for i in rec_ids if i in self.pos]
        if not rec_ids:
            return []
        s = self.codes([self.pos[i] for i in rec_ids]) @ self.readout(text)
        return sorted(zip(rec_ids, (float(x) for x in s)), key=lambda x: -x[1])

    # ---------------------------------------------------------------- forget
    def forget(self, rec_ids, all_records: list[dict]) -> None:
        """Really delete: rebuild the index from the remaining records, in the order given (each needs "id").
        Nothing of the forgotten records stays in W or in the unit counts."""
        drop = set(rec_ids)
        self.W[:] = 0
        self.n_use[:] = 0
        self.ids, self.pos = [], {}
        for r in all_records:
            if r["id"] not in drop:
                self.add(r["id"], r)

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | os.PathLike) -> None:
        """Atomic save of everything the index learned (codes are recomputed from positions)."""
        path = str(path)
        tmp = path + ".tmp.npz"
        meta = {"dim": self.dim, "block_orthogonal": self.block_orthogonal, "seed": self.seed, "K": K, "KS": KS,
                "gamma": GAMMA, "ids": self.ids}
        with open(tmp, "wb") as fh:
            np.savez(fh, W=self.W, n_use=self.n_use, meta=np.array(json.dumps(meta, ensure_ascii=False)))
        os.replace(tmp, path)

    def load(self, path: str | os.PathLike) -> "NetIndex":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["dim"] != self.dim:
            raise ValueError(f"index file has dim {meta['dim']}, this index {self.dim}")
        self.block_orthogonal, self.seed = meta["block_orthogonal"], meta["seed"]
        self._blocks = {}
        self.W, self.n_use = z["W"].copy(), z["n_use"].copy()
        self.ids = [tuple(i) if isinstance(i, list) else i for i in meta["ids"]]
        self.pos = {i: p for p, i in enumerate(self.ids)}
        return self
