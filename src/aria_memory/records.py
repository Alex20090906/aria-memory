"""Records and how the reader sees them.

A record is a plain dict: {"thing", "aspect", "value", "day", "heard", "src"} plus, once stored, "id".
    aspect   one of prompts.SLOTS (location, time, status, preference, plan, relation, attribute, value)
    heard    the sentence it was heard in (night rows: their source sentence, or "" when the listener named none)
    src      "owner" | "sleep" | "uncertain" (speaker not confirmed) | "self" (the assistant's own thought)
"""
from __future__ import annotations

from .prompts import ASPECT_ZH

NO_SOURCE = "(sleep)"          # E49 showed night rows without a source sentence like this


def aspect_zh(aspect: str) -> str:
    return ASPECT_ZH.get(aspect, aspect)


def heard_of(rec: dict) -> str:
    return rec.get("heard") or NO_SOURCE


def key_text(rec: dict) -> str:
    """What the index hashes for a record (E49: entity, slot name, value, heard sentence)."""
    return f"{rec['thing']} {aspect_zh(rec['aspect'])} {rec['value']} {heard_of(rec)}"


def row_text(rec: dict) -> str:
    src = rec.get("src", "owner")
    if src == "uncertain":
        tail = f"（原話：{heard_of(rec)}；說話的人不確定，不一定是使用者）"
    elif src == "self":
        tail = "（自己想的，不是使用者說的）"
    else:
        tail = f"（原話：{heard_of(rec)}）"
    return f"{rec['thing']}｜{aspect_zh(rec['aspect'])}｜{rec['value']}{tail}"


def order(records: list[dict]) -> list[dict]:
    """Oldest first, as E49 listed the top 5 (by day, then by id or arrival)."""
    return [r for _, r in sorted(enumerate(records), key=lambda x: (x[1].get("day", 0), x[1].get("id", x[0])))]


def render_records(records: list[dict], *, now_day: int | None = None) -> str:
    """The numbered block the reader sees: `N. 第{day}天｜thing｜aspect｜value（原話：…）`, oldest first.
    now_day is accepted for callers that want to add 今天是第N天 themselves; it is not printed here."""
    return "\n".join(f"{n + 1}. 第{r.get('day', 0)}天｜{row_text(r)}" for n, r in enumerate(order(records)))
