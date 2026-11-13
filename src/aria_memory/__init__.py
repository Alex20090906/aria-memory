"""Aria's memory system (research E49, 2026-11-06) as a pure library for a voice assistant.

No bus, no HTTP, no threads, no file paths of its own: the caller owns storage, scheduling and the LLM call
`ask(messages, max_tokens) -> str` (async, text without thinking).

    Listener / listen        one owner sentence -> records (TOPIC, staged commit, H29 dates, H40 cleaning)
    render_records / read    how the reader sees records; retrieve-then-read with the sound-twin second read
    Vocab / NetIndex         the thinker model's layer-0 word vectors + pinyin -> hippocampal index in fast weights
    CharVocab                a model-free vocabulary for the index (characters and words, hashed)
    relisten / supersede     the night pass (H30; H42b strict marks)
    Gate / IdleWork          the always-on gate: energy pool, four fuses, idle steps, proactive cap
    ReflexGRU                the research reflex model in numpy (parity only, not wired)
    eval / replay            the 21-day scripts, strict scoring, wake-word coverage, replays
"""
from .gate import Gate, IdleWork
from .index import NetIndex
from .listen import Listener, ListenResult, listen
from .prompts import (ASPECT_ZH, LISTEN_SYS, PLAN_SYS, READ_RULES, READ_SYS, RELISTEN_SYS, SLOTS, STAGE_SYS,
                      SUPERSEDE_SYS)
from .read import read, sound_twins
from .records import key_text, render_records, row_text
from .reflex_gru import ReflexGRU
from .rules import clean, fix_reldate, resolve
from .sleep import relisten, supersede
from .vocab import CharVocab, Vocab

__all__ = ["ASPECT_ZH", "CharVocab", "Gate", "IdleWork", "LISTEN_SYS", "Listener", "ListenResult", "NetIndex", "PLAN_SYS",
           "READ_RULES", "READ_SYS", "RELISTEN_SYS", "ReflexGRU", "SLOTS", "STAGE_SYS", "SUPERSEDE_SYS", "Vocab",
           "clean", "fix_reldate", "key_text", "listen", "read", "relisten", "render_records", "resolve", "row_text",
           "sound_twins", "supersede"]
