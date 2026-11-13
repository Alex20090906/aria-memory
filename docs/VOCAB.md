# Building a model vocabulary for the index

The index works without any model (`CharVocab`). The research used the thinker model's own vocabulary: its tokenizer
plus, for every token, 4 "meaning units" computed from the token's layer-0 word vector. With it the index found the
right record in the top 5 for 35/38 and 34/38 questions, against 34/38 and 34/38 with `CharVocab`
(chapter 4.3 of [research/](research/04-index-and-record-log.md)).

The vocabulary is derived from the model's weights, so the model's licence applies to it. It is not shipped.

## Inputs

| File | What | Research model (Qwen3.8-Flash-Next) |
|---|---|---|
| `token_embd.npy` | the model's input embedding table (`token_embd.weight` in a GGUF file), dequantised, shape `[vocab, dim]`, fp16 is fine | 248,320 × 2,560, about 1.2 GB |
| `vocab.json`, `merges.txt` | the byte-level BPE vocabulary and merges from the GGUF metadata (`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`) | |

Any GGUF reader can produce these (for example the `gguf` Python package from llama.cpp). The research extracted
them with tools on the research machine that are not part of this repository, so the extraction step is described,
not provided.

The tokenizer uses llama.cpp's `qwen35` pre-tokenizer pattern (`PRE_PATTERN` in `src/aria_memory/vocab.py`). For a
model with a different pre-tokenizer, change that pattern.

## Build (CPU)

```bash
pip install -e ".[vocab]"
python -m aria_memory.vocab build --table token_embd.npy --vocab vocab.json --merges merges.txt --out data/aria_vocab
```

What it does: normalises every row with per-dimension statistics from 4,000 sampled rows, multiplies by a fixed random
projection (seed 1) into 32,768 units, and keeps each token's 4 strongest units; tokens without word content
(punctuation, spaces, partial bytes) get none. Output: `tokenizer.json`, `token_units.npy` (int32, `[vocab, 4]`,
3.8 MB for the research model) and `meta.json`.

## Use

```python
from aria_memory import NetIndex, Vocab
index = NetIndex(Vocab.load("data/aria_vocab"))
```

Tests pick it up from `data/aria_vocab/` or from the folder in the `ARIA_VOCAB` environment variable; the
quickstart only from `ARIA_VOCAB`:

```bash
ARIA_VOCAB=data/aria_vocab pytest -q            # 4 more tests, including the research parity checks
ARIA_VOCAB=data/aria_vocab python examples/quickstart.py
```

The parity tests expect the research model's vocabulary; with another model's table they will report different
numbers, which is expected.
