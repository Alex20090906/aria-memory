"""Listening: one owner sentence in, records out (E49 = the E43 listening of llm_fastmem/zh_v3.py).

Listener keeps what E49 kept between sentences:
  - TOPIC, the thing being talked about, so 它/那個/裡面 resolve to a concrete thing;
  - staged commit: a fact waits STAGE_TTL more owner sentences before it leaves as a record, so 改口 (correction) and
    補充 (detail) replace or complete it first.
Relative dates are fixed by rule (H29) and non-facts dropped (H40) before a record leaves.

`ask(messages, max_tokens) -> str` is the caller's async LLM call (no thinking text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompts import LISTEN_SHOTS, LISTEN_SYS, SLOTS, STAGE_SYS, WEEKDAY_ZH
from .rules import clean, fix_reldate, strip_suffix, weekday_of

STAGE_TTL = 2
CTX_LINES = 5                       # previous lines shown as 前文 (E49: the last 6 lines including the current one)
MAX_TOKENS = 160
STAGE_RE = re.compile(r"^(改口|補充)\s*#\s*(\d+)\s*\|\s*(.+)$")


@dataclass
class ListenResult:
    records: list[dict] = field(default_factory=list)      # facts that leave staging now
    questions: list[tuple[str, str]] = field(default_factory=list)   # (thing, aspect) the sentence asks about
    topic: str = "（無）"
    raw: str = ""                                           # the listener's text, for logs


def parse(lines, ops: list | None = None):
    """(facts, questions, topic) from the listener's lines. Question lines are kept with any aspect (H32: the
    retrieve-then-read answerer does not need the slot)."""
    facts, qs, topic = [], [], None
    for line in lines:
        line = line.strip()
        m = STAGE_RE.match(line)
        if m and ops is not None:
            ops.append((m.group(1), int(m.group(2)), m.group(3).strip()))
            continue
        if not line or line.upper().startswith("NONE"):
            continue
        if line.upper().startswith("TOPIC"):
            topic = line[5:].strip(" ：:")
            continue
        if line.startswith("Q") and line.count("|") == 1:
            e, s = [x.strip() for x in line[1:].split("|")]
            if e and (strip_suffix(e), s) not in qs:
                qs.append((strip_suffix(e), s))
        elif line.count("|") == 2:
            e, s, v = [x.strip() for x in line.split("|")]
            if s in SLOTS and e and v:
                facts.append((strip_suffix(e), s, v))
    return facts[:3], qs[:3], topic


def day_line(day: int, weekday: int | None = None) -> str:
    return f"今天是第{day}天（星期{WEEKDAY_ZH[weekday_of(day, weekday) - 1]}）。"


def _ctx_line(item) -> str:
    if isinstance(item, (tuple, list)):
        text, owner = item[0], bool(item[1])
    else:
        text, owner = str(item), True
    return f"{'[使用者]' if owner else '[其他人]'} {text}"


def _messages(system: str, user: str, shots=LISTEN_SHOTS) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    for a, b in shots:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": user})
    return msgs


def _record(thing, aspect, value, day, heard, src) -> dict | None:
    v, keep = clean(value)
    if not keep:
        return None
    return {"thing": thing, "aspect": aspect, "value": v, "day": day, "heard": heard, "src": src}


class Listener:
    """Stateful E49 listener. Persist with state() / Listener.from_state()."""

    def __init__(self, ttl: int = STAGE_TTL):
        self.ttl = ttl
        self.topic = "（無）"
        self.buf: list[dict] = []          # staged facts: thing, aspect, value, day, heard, src, ttl

    # ------------------------------------------------------------------ persistence (JSON-safe)
    def state(self) -> dict:
        return {"ttl": self.ttl, "topic": self.topic, "buf": [dict(x) for x in self.buf]}

    @classmethod
    def from_state(cls, d: dict) -> "Listener":
        lis = cls(ttl=d.get("ttl", STAGE_TTL))
        lis.topic = d.get("topic", "（無）")
        lis.buf = [dict(x) for x in d.get("buf", [])]
        return lis

    # ------------------------------------------------------------------ listening
    def prompt(self, sentence: str, *, day: int, weekday: int | None = None, recent=()) -> list[dict]:
        ctx = [_ctx_line(x) for x in list(recent)[-CTX_LINES:]]
        staged = "\n暫存：\n" + ("\n".join(f"#{i + 1} {it['thing']} | {it['aspect']} | {it['value']}"
                                          for i, it in enumerate(self.buf)) or "（無）")
        user = (day_line(day, weekday) + f"\nTOPIC：{self.topic}" + "\n前文：\n" + ("\n".join(ctx) or "（無）")
                + staged + "\n剛聽到的一句：\n[使用者] " + sentence)
        return _messages(LISTEN_SYS + STAGE_SYS, user)

    async def hear(self, ask, sentence: str, *, day: int, weekday: int | None = None, recent=(),
                   src: str = "owner") -> ListenResult:
        """One owner sentence. Returns the records leaving staging now, the questions asked, and the topic."""
        raw = (await ask(self.prompt(sentence, day=day, weekday=weekday, recent=recent), MAX_TOKENS)) or ""
        ops: list = []
        facts, qs, tp = parse(raw.strip().splitlines(), ops)
        facts = [(e, s, fix_reldate(s, v, sentence, day, weekday)) for e, s, v in facts]
        if tp:
            self.topic = tp
        for _kind, idx, val in ops:                      # 改口 / 補充 of a staged fact
            if 1 <= idx <= len(self.buf):
                it = self.buf[idx - 1]
                it["value"] = fix_reldate(it["aspect"], val, sentence, day, weekday)
                it["ttl"] = self.ttl                     # a changed fact waits again
        for it in self.buf:
            it["ttl"] -= 1
        out = [x for x in self.buf if x["ttl"] < 0]
        self.buf = [x for x in self.buf if x["ttl"] >= 0]
        for e, s, v in facts:
            same = [x for x in self.buf if x["thing"] == e and x["aspect"] == s]
            if same:
                same[-1]["value"] = v
                same[-1]["ttl"] = self.ttl
            else:
                self.buf.append({"thing": e, "aspect": s, "value": v, "day": day, "heard": sentence, "src": src,
                                 "ttl": self.ttl})
        records = [r for r in (_record(x["thing"], x["aspect"], x["value"], x["day"], x["heard"], x["src"])
                               for x in out) if r]
        return ListenResult(records=records, questions=qs, topic=self.topic, raw=raw)

    def flush(self) -> list[dict]:
        """End of day: everything still staged leaves as records; the topic resets."""
        out = [r for r in (_record(x["thing"], x["aspect"], x["value"], x["day"], x["heard"], x["src"])
                           for x in self.buf) if r]
        self.buf, self.topic = [], "（無）"
        return out

    def staged(self) -> list[dict]:
        """Facts heard but not yet stored (answerable as working memory)."""
        return [{k: v for k, v in x.items() if k != "ttl"} for x in self.buf]


async def listen(ask, sentence: str, *, day: int, weekday: int | None = None, recent=(), topic: str = "（無）",
                 src: str = "owner") -> list[dict]:
    """Stateless convenience: one sentence, records at once (no staging, so not E49's exact behaviour)."""
    ctx = [_ctx_line(x) for x in list(recent)[-CTX_LINES:]]
    user = (day_line(day, weekday) + f"\nTOPIC：{topic}" + "\n前文：\n" + ("\n".join(ctx) or "（無）")
            + "\n剛聽到的一句：\n[使用者] " + sentence)
    raw = (await ask(_messages(LISTEN_SYS, user), MAX_TOKENS)) or ""
    facts, _qs, _tp = parse(raw.strip().splitlines())
    out = []
    for e, s, v in facts:
        r = _record(e, s, fix_reldate(s, v, sentence, day, weekday), day, sentence, src)
        if r:
            out.append(r)
    return out
