"""Evaluation on the research's 21-day scripts (two sets: "set_old" and "set2").

A script set folder (tests/fixtures/aria/<set>/) holds:
    zh_scripts.json     facts (90) and questions (72: 51 told, 21 never told or only similar) with expected answers
    timeline.json       per day: every spoken segment with speaker (owner, phone_owner, family, tv, visitor ...),
                        start/end seconds in that day's audio, and text
    asr.json            per day: what speech recognition heard (segments with start/end, text, voice-ID owner flag)
    listen_cache.json   E43's cached LLM listening of exactly these segments (writes, answers), which E49 replays

score() is the strict rescoring of the always-on runs (alwayson/rescore_strict.py): numbers and weekdays normalised,
at least 60% character overlap with the expected answer, and a different 第N天 than expected is wrong.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

FILLER = set("的了在是我你他她它那個這個就啊啦喔欸嗎呢吧齁上面裡面下面一下有沒")
UNKNOWN = re.compile(r"(不記得|不知道|沒有印象|沒聽你說|沒講過|沒說過|忘了|不確定|don't|do not)")
DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
UNIT = {"十": 10, "百": 100, "千": 1000}
DAY = re.compile(r"第\s*(\d+)\s*天")
WAKE_OLD = "Aria"         # the assistant's name in the research scripts


def chars(s: str) -> set[str]:
    return {c for c in re.sub(r"\W", "", s or "") if c not in FILLER}


def overlap(a: str, b: str) -> float:
    ca = chars(a)
    return len(ca & chars(b)) / max(1, len(ca))


def _cn_to_int(s: str) -> int:
    total, cur = 0, 0
    for ch in s:
        if ch in DIG:
            cur = DIG[ch]
        elif ch in UNIT:
            total += (cur or 1) * UNIT[ch]
            cur = 0
    return total + cur


def norm(s: str) -> str:
    """Spelling only: Chinese numerals -> digits, 禮拜/星期/周 -> 週, 塊 -> 元, full-width digits -> ASCII."""
    s = s or ""
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = re.sub(r"(禮拜|星期|周)", "週", s)
    s = s.replace("塊", "元").replace("本週", "這週").replace("這禮拜", "這週")
    s = re.sub(r"週([一二三四五六日天])", lambda m: "週" + "一二三四五六日日"["一二三四五六日天".index(m.group(1))] + "§", s)
    s = re.sub(r"[零〇一二兩三四五六七八九十百千]+", lambda m: str(_cn_to_int(m.group(0))), s)
    s = re.sub(r"週(\d)§", lambda m: "週" + "一二三四五六日"[int(m.group(1)) - 1] if 1 <= int(m.group(1)) <= 7
               else m.group(0), s)
    return s.replace("§", "")


def right(expect: str, tail: str) -> bool:
    if overlap(norm(expect), norm(tail)) < 0.6:
        return False
    de, dt = DAY.findall(expect), DAY.findall(tail)
    return not (de and dt and de[0] != dt[0])


def rename_wake(text: str, wake: str | None = None) -> str:
    """The research scripts call the assistant Aria. With another wake word (default: prompts.NAME) every "Aria" in a
    line becomes that word, spaces around it dropped; with "Aria" the text is returned as it is."""
    from .prompts import NAME

    wake = wake or NAME
    if wake == WAKE_OLD:
        return text or ""
    return re.sub(r" ?Aria ?", wake, text or "")


# ---------------------------------------------------------------------------------------------- loading
def _load(p: Path):
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(p.read_text(encoding="utf-8"))


class ScriptSet:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.name = folder.name
        self.sc = _load(folder / "zh_scripts.json")
        self.timeline = {d["day"]: d for d in _load(folder / "timeline.json")}
        self.asr = _load(folder / "asr.json")
        cache = folder / "listen_cache.json"
        self.cache = _load(cache) if cache.exists() else None

    @property
    def questions(self) -> list[dict]:
        return self.sc["questions"]

    @property
    def facts(self) -> list[dict]:
        return self.sc["facts"]

    def segment_of(self, q: dict) -> dict | None:
        return next((g for g in self.timeline[q["day"]]["segments"] if g["text"] == q["text"]), None)

    def lines(self) -> list[dict]:
        """Every scripted line: day, start/end (seconds into that day's audio), speaker, addressee, text (Aria
        renamed to the wake word, see rename_wake), and the question it is (id = index in questions, expected, stale) if any."""
        qidx = {(q["day"], q["text"]): i for i, q in enumerate(self.questions)}
        out = []
        for day in sorted(self.timeline):
            for g in self.timeline[day]["segments"]:
                i = qidx.get((day, g["text"]))
                q = self.questions[i] if i is not None else None
                out.append({"day": day, "start": g["start"], "end": g["end"], "speaker": g["speaker"],
                            "to": g.get("to"), "text": rename_wake(g["text"]), "text_orig": g["text"],
                            "question": None if q is None else {"id": i, "expect": q["expect"], "stale": q["stale"],
                                                                "thing": q["entity"], "aspect": q["slot"]}})
        return out

    def heard(self) -> list[dict]:
        """What speech recognition produced: day, start, end, text (Aria renamed to the wake word), owner flag of voice ID
        ("owner" | "other" | "unsure"), and the scripted speaker of the most-overlapping line."""
        out = []
        for d in self.asr:
            segs = self.timeline[d["day"]]["segments"]
            for h in d["heard"]:
                best = max(segs, key=lambda g: min(g["end"], h["end"]) - max(g["start"], h["start"]), default=None)
                ov = min(best["end"], h["end"]) - max(best["start"], h["start"]) if best else 0
                out.append({"day": d["day"], "start": h["start"], "end": h["end"], "text": rename_wake(h["text"]),
                            "text_orig": h["text"], "voice": h["owner"],
                            "speaker": best["speaker"] if best and ov > 0 else None})
        return out


# ---------------------------------------------------------------------------------------------- scoring
def score(answers: list[dict], script: ScriptSet) -> dict:
    """Strict rescoring. Each answer: {"day", "start", "end" (seconds of the heard question), "thing" (what the
    listener said was asked), "passed" (an answer was given), "value" (the answer's content)}.
    For each scripted question the answer overlapping its segment by > 0.3 s is used (several questions can share a
    heard segment: the one whose thing matches best)."""
    hit = wrong = held = never = told = 0
    rows = []
    for i, q in enumerate(script.questions):
        g = script.segment_of(q)
        cand = [a for a in answers if g and a["day"] == q["day"] and a.get("start") is not None
                and min(a["end"], g["end"]) - max(a["start"], g["start"]) > 0.3]
        cand.sort(key=lambda a: -overlap(q["entity"], a.get("thing") or ""))
        a = cand[0] if cand else None
        if q["expect"] is None:
            never += 1
            kept = a is None or not a["passed"]
            held += kept
            rows.append({"id": i, "told": False, "kept": kept, "answer": a})
        else:
            told += 1
            ok = bool(a and a["passed"] and right(q["expect"], a.get("value") or ""))
            hit += ok
            wrong += bool(a and a["passed"] and not ok)
            rows.append({"id": i, "told": True, "right": ok, "answer": a})
    return {"told_right": hit, "told_wrong": wrong, "told_total": told, "untold_held": held, "untold_total": never,
            "rows": rows}
