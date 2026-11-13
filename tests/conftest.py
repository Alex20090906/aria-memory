"""Shared test helpers.

small_vocab(): the model vocabulary sample (tests/fixtures/vocab_small, 1,595 character tokens) when present, else
the model-free CharVocab. The sample is derived from the thinker model's layer-0 table and is not shipped by default;
see docs/REVIEW.md. Tests that need the thinker model's tokenizer are skipped without it.
"""
from pathlib import Path

from aria_memory.vocab import CharVocab, Vocab

FIX = Path(__file__).parent / "fixtures"
LLM_SMALL = (FIX / "vocab_small" / "token_units.npy").exists()


def small_vocab():
    return Vocab.load(FIX / "vocab_small") if LLM_SMALL else CharVocab()
