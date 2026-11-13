"""the model vocabulary for the memory index: its tokenizer and, for every token, the 4 "meaning units" its layer-0
word vector lands on.

The index key of a text (see index.py) uses each model token's layer-0 vector (row of token_embd), normalised and
multiplied by a fixed random projection (seed 1, 2560 x 32768); the token's 4 strongest projected units are its
meaning units. That is the only use of the vectors, so the units are precomputed once for the whole vocabulary
(248,320 x 4 int32, about 4 MB) and the 1.2 GB table is not needed at runtime.

A vocabulary folder holds:
    tokenizer.json     the model's byte-level BPE (vocab + merges from its GGUF, llama.cpp's qwen35 pre-tokenizer)
    token_units.npy    int32 [vocab, 4]; -1 rows are tokens with no word content (punctuation, spaces, partial bytes)
    meta.json          what was built from what

Build it once on CPU (E49 used the same projection and normalisation):
    python -m aria_memory.vocab build --table token_embd_f16.npy --vocab vocab.json --merges merges.txt --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

H = 32768            # meaning bank size (units 0..H-1); the sound bank is H..2H-1
KT = 4               # meaning units per token
SEED = 1             # projection seed (E49)
N_SAMPLE = 4000      # rows used for the per-dimension normalisation (E49)
# llama.cpp src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35 (the model's GGUF names pre = "qwen35")
PRE_PATTERN = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
               r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
WORDISH = re.compile(r"[\w一-鿿]")


def build_tokenizer(vocab_json: str | os.PathLike, merges_txt: str | os.PathLike):
    """The thinker model's tokenizer from its vocab and merges (byte-level BPE, qwen35 pre-tokenizer, NFC as in Qwen2)."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

    vocab = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
    merges = []
    for line in Path(merges_txt).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#version"):
            continue
        a, b = line.split(" ")
        merges.append((a, b))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRE_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def _projection(dim: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return (rng.standard_normal((dim, H)) / np.sqrt(dim)).astype(np.float32)


def _rownorm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)


def sample_stats(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and spread over the same 4000 sampled rows E49 used (rng seed 1)."""
    rng = np.random.default_rng(SEED)
    s = _rownorm(np.asarray(table[np.sort(rng.integers(0, table.shape[0], N_SAMPLE))]))
    return s.mean(0), s.std(0) + 1e-3


def units_of_rows(rows: np.ndarray, mu: np.ndarray, sd: np.ndarray, P: np.ndarray) -> np.ndarray:
    """The KT meaning units of each row (int32 [n, KT]); same arithmetic as E49's NetIndex._token_units."""
    x = (_rownorm(rows) - mu) / sd
    s = x @ P
    return np.argpartition(-s, KT, axis=1)[:, :KT].astype(np.int32)


def build_units(table: np.ndarray, keep: np.ndarray, chunk: int = 2048, log=print) -> np.ndarray:
    mu, sd = sample_stats(table)
    P = _projection(table.shape[1])
    out = np.full((table.shape[0], KT), -1, np.int32)
    t0 = time.time()
    for a in range(0, table.shape[0], chunk):
        b = min(a + chunk, table.shape[0])
        u = units_of_rows(np.asarray(table[a:b]), mu, sd, P)
        k = keep[a:b]
        out[a:b][k] = u[k]
        if log and (a // chunk) % 20 == 0:
            log(f"  rows {b}/{table.shape[0]}  {time.time() - t0:.0f}s")
    return out


def wordish_mask(tok, n: int) -> np.ndarray:
    """E49 skipped tokens whose decoded piece has no word character (spaces, punctuation, partial UTF-8 bytes)."""
    keep = np.zeros(n, bool)
    for i in range(n):
        piece = tok.decode([i]).strip()
        keep[i] = bool(piece) and bool(WORDISH.search(piece))
    return keep


class Vocab:
    """Tokenizer plus precomputed meaning units. Load with Vocab.load(folder)."""

    def __init__(self, tokenizer, units: np.ndarray, meta: dict | None = None):
        self.tokenizer = tokenizer
        self.table = np.asarray(units, np.int32)
        self.meta = meta or {}

    @classmethod
    def load(cls, folder: str | os.PathLike) -> "Vocab":
        from tokenizers import Tokenizer

        folder = Path(folder)
        tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        units = np.load(folder / "token_units.npy")
        meta_p = folder / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        return cls(tok, units, meta)

    def save(self, folder: str | os.PathLike) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(folder / "tokenizer.json"))
        np.save(folder / "token_units.npy", self.table)
        (folder / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text or "", add_special_tokens=False).ids

    def units(self, token_id: int) -> np.ndarray:
        """Meaning units of a token; empty for tokens without word content or outside the table."""
        if not 0 <= token_id < len(self.table):
            return np.zeros(0, np.int32)
        u = self.table[token_id]
        return u if u[0] >= 0 else np.zeros(0, np.int32)


class CharVocab:
    """A vocabulary that needs no model (not part of E49; added for the open-source release).

    Every CJK character and every run of letters or digits is a token, and its KT meaning units are hashed from the
    token itself. Meaning units then match only the same characters or words: no synonyms, no shared meaning between
    different words. The sound bank in index.py still matches characters that sound alike. Use it to try the index
    without the model's layer-0 table, or when the reader is any other LLM.
    """

    TOKEN = re.compile(r"[一-鿿]|[a-z]+|\d+")

    def __init__(self):
        self.meta = {"source": "characters and words, hashed (no model)", "KT": KT, "H": H}
        self._ids: dict[str, int] = {}
        self._tokens: list[str] = []

    def token_ids(self, text: str) -> list[int]:
        out = []
        for m in self.TOKEN.finditer(unicodedata.normalize("NFKC", text or "").lower()):
            s = m.group(0)
            i = self._ids.get(s)
            if i is None:
                i = self._ids[s] = len(self._tokens)
                self._tokens.append(s)
            out.append(i)
        return out

    def units(self, token_id: int) -> np.ndarray:
        if not 0 <= token_id < len(self._tokens):
            return np.zeros(0, np.int32)
        s = self._tokens[token_id]
        return np.array([int(hashlib.md5(f"m:{s}#{j}".encode()).hexdigest()[:8], 16) % H for j in range(KT)], np.int32)


def build(table_path, vocab_json, merges_txt, out_dir, log=print) -> Vocab:
    tok = build_tokenizer(vocab_json, merges_txt)
    table = np.load(table_path, mmap_mode="r")
    n = table.shape[0]
    log(f"table {table.shape} {table.dtype}; tokenizer vocab {tok.get_vocab_size()}")
    keep = wordish_mask(tok, min(n, tok.get_vocab_size()))
    keep = np.concatenate([keep, np.zeros(n - len(keep), bool)])
    units = build_units(table, keep, log=log)
    meta = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": int(n), "dim": int(table.shape[1]),
            "tokens_with_units": int(keep.sum()), "KT": KT, "H": H, "seed": SEED, "n_sample": N_SAMPLE,
            "pre_pattern": "llama.cpp qwen35", "source": "Qwen3.8-Flash-Next token_embd (layer 0)"}
    v = Vocab(tok, units, meta)
    v.save(out_dir)
    log(f"saved {out_dir}: {keep.sum()} tokens with units")
    return v


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m aria_memory.vocab")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a vocabulary folder from the model's layer-0 table")
    b.add_argument("--table", required=True, help="token_embd as .npy [vocab, dim] (fp16 is fine)")
    b.add_argument("--vocab", required=True, help="vocab.json from the GGUF")
    b.add_argument("--merges", required=True, help="merges.txt from the GGUF")
    b.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "build":
        build(a.table, a.vocab, a.merges, a.out)


if __name__ == "__main__":
    main()
