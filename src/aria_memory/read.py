"""Reading: the index's top records go to the reader (an LLM), which names the one about exactly what was asked or says it does
not know (E49 = retrieve-then-read H28 + the sound-twin second read H31b, gate off E47).

    out = await read(ask, question, top, now_day=day)
    out = {"status": "answer" | "unknown", "reply": str, "rec_id": id | None, "quote": heard sentence | None,
           "record": dict | None, "mode": "read" | "read_phon" | "none"}

`top` is the records the index returned (k = 5), each with "id". They are listed oldest first.
"""
from __future__ import annotations

import re

from .index import is_cjk, syllable
from .prompts import READ_SYS
from .records import order, render_records

UNKNOWN = re.compile(r"(不記得|不知道|沒有印象|沒聽你說|沒講過|沒說過|忘了|不確定|don't|do not)")
PICK = re.compile(r"編號\s*(\d+)")
MAX_TOKENS = 80


def sound_twins(question: str, records: list[dict]) -> list[tuple[dict, str]]:
    """(record, word in the question) for records whose thing sounds exactly like a different word in the question
    (4+ characters: one syllable may differ). Speech recognition wrote the wrong characters."""
    qc = [(ch, syllable(ch)) for ch in question if is_cjk(ch)]
    out = []
    for r in records:
        e = r["thing"]
        if len(e) < 2 or not all(is_cjk(ch) for ch in e):
            continue
        es = [syllable(ch) for ch in e]
        n = len(es)
        for a in range(len(qc) - n + 1):
            win = qc[a:a + n]
            diff = sum(x[1] != y for x, y in zip(win, es))
            sub = "".join(x[0] for x in win)
            if sub != e and (diff == 0 or (n >= 4 and diff == 1)):
                out.append((r, sub))
                break
    return out


def read_prompt(question: str, records: list[dict], *, now_day: int, hint: str | None = None,
                note: str | None = None) -> list[dict]:
    user = f"今天是第{now_day}天。\n使用者問：{question}\n"
    if hint:
        user += f"（這一題問的重點可能是：{hint}，可能聽錯）\n"
    if note:
        user += f"（注意：{note}，是語音辨識聽錯，指的是同一個東西。還是要問的那一方面對得上才回答。）\n"
    user += "記憶紀錄：\n" + render_records(records, now_day=now_day)
    return [{"role": "system", "content": READ_SYS}, {"role": "user", "content": user}]


def _pick(text: str, n: int) -> tuple[int | None, str]:
    lines = (text or "").strip().splitlines()
    first = lines[0] if lines else ""
    reply = " ".join(lines[1:]).strip()
    m = PICK.search(first)
    if m and not UNKNOWN.search(first) and 1 <= int(m.group(1)) <= n:
        return int(m.group(1)) - 1, reply
    return None, reply


def _answer(rec: dict, reply: str, mode: str) -> dict:
    return {"status": "answer", "reply": reply, "rec_id": rec.get("id"), "quote": rec.get("heard") or None,
            "record": rec, "mode": mode}


async def read(ask, question: str, top: list[dict], *, now_day: int, hint: str | None = None,
               phon: bool = True) -> dict:
    none = {"status": "unknown", "reply": "我不記得了。", "rec_id": None, "quote": None, "record": None, "mode": "none"}
    if not top:
        return none
    recs = order(top)
    i, reply = _pick(await ask(read_prompt(question, recs, now_day=now_day, hint=hint), MAX_TOKENS), len(recs))
    if i is not None:
        return _answer(recs[i], reply, "read")
    if phon:
        twins = sound_twins(question, recs)
        if twins:
            note = "；".join(f"問句裡的「{sub}」和紀錄裡的「{r['thing']}」唸起來一樣" for r, sub in twins)
            trecs = [r for r, _ in twins]
            j, reply2 = _pick(await ask(read_prompt(question, trecs, now_day=now_day, note=note), MAX_TOKENS),
                              len(trecs))
            if j is not None:
                return _answer(trecs[j], reply2, "read_phon")
            reply = reply2 or reply
    return dict(none, reply=reply or none["reply"])
