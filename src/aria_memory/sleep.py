"""Sleep: the night pass over the day.

relisten (H30, part of E49): The LLM re-reads everything the owner said today and writes the final version of each fact
(corrections resolved, pronouns filled in), each row naming the sentence it rests on. Labels already known are skipped
(E49 skipped a night row when the label was already familiar).

supersede (H42b, strict; ported from the offline test, NOT part of the E49 final run, no measurable gain over 21 days):
every record of today looks up the 3 older records the index ranks highest for it, keeps those of the same aspect and
the same thing label, and the LLM judges which of them it replaces (same entity, same aspect, new content). A new value
that is not of its aspect's kind (a location that names no place, a time with no day or clock time) replaces nothing.
Marks only: records are never edited, superseded ones are just left out of later reading.
"""
from __future__ import annotations

import re

from .listen import day_line
from .prompts import ASPECT_ZH, RELISTEN_SYS, SLOTS, SUPERSEDE_SYS
from .records import key_text, row_text
from .rules import DAYN, clean, fix_reldate, resolve_value, strip_suffix

RELISTEN_TOKENS = 600
SUPERSEDE_TOKENS = 20
ZH_ASPECT = {v: k for k, v in ASPECT_ZH.items()}

# Two guards on supersede, added after the P4 run with the LLM server (set_old: 3 wrong marks of 21). A record replaces only
# another record of the same thing (the label, not just a similar one: 牙醫 is not 洗牙, 咖啡 is not 飲料), and only
# with a value of the aspect's kind (a location names a place: 拿去 does not; a time names a day or a clock time).
PLACE = re.compile(r"[上下裡內中旁邊櫃箱屜桌房包袋架盒門台臺室廳車櫥間層角底頂側前後左右座縫隙口店樓院庫處]|公司|家")
TIMEISH = re.compile(r"第\s*\d+\s*天|\d+\s*[點:：]|[週周]|禮拜|星期|\d+\s*[月號日]|早上|上午|中午|下午|傍晚|晚上|凌晨|"
                     r"今天|明天|後天|大後天|月底|月初|年底")
KIND = {"location": PLACE, "time": TIMEISH}


def same_thing(a: str, b: str) -> bool:
    import unicodedata
    key = lambda s: re.sub(r"\s", "", strip_suffix(unicodedata.normalize("NFKC", s or "")))
    return key(a) == key(b)


def value_of_kind(aspect: str, value: str) -> bool:
    """A location value names a place, a time value names a day or a clock time; other aspects are not checked."""
    pattern = KIND.get(aspect)
    return pattern is None or bool(pattern.search(clean(value)[0]))


def relisten_prompt(day_sentences: list[str], *, day: int, weekday: int | None = None) -> list[dict]:
    user = day_line(day, weekday) + "\n今天使用者說過的話：\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(day_sentences))
    return [{"role": "system", "content": RELISTEN_SYS}, {"role": "user", "content": user}]


def parse_relisten(text: str, day_sentences: list[str], *, day: int, weekday: int | None = None) -> list[dict]:
    out = []
    for line in (text or "").splitlines():
        parts = [x.strip() for x in re.split(r"[｜|]", line)]
        if len(parts) not in (3, 4):
            continue
        e, s, v = parts[:3]
        s = ZH_ASPECT.get(s, s if s in SLOTS else None)
        if not (s and e and v):
            continue
        src = None
        if len(parts) == 4:
            m = re.search(r"\d+", parts[3])
            if m and 1 <= int(m.group()) <= len(day_sentences):
                src = day_sentences[int(m.group()) - 1]
        if s in ("time", "plan"):
            v2 = resolve_value(v, day, weekday)
            if v2 == v and src and DAYN.search(v):
                v2 = fix_reldate(s, v, src, day, weekday)
            v = v2
        v, keep = clean(v)
        if not keep:
            continue
        out.append({"thing": strip_suffix(e), "aspect": s, "value": v, "day": day, "heard": src or "", "src": "sleep"})
    return out


async def relisten(ask, day_sentences: list[str], *, day: int, weekday: int | None = None,
                   known_labels=()) -> list[dict]:
    """Night records for today. `known_labels`: (thing, aspect) pairs already stored; those are skipped."""
    if not day_sentences:
        return []
    text = await ask(relisten_prompt(day_sentences, day=day, weekday=weekday), RELISTEN_TOKENS)
    known = {tuple(x) for x in known_labels}
    out, seen = [], set()
    for r in parse_relisten(text, day_sentences, day=day, weekday=weekday):
        lab = (r["thing"], r["aspect"])
        if lab in known or lab in seen:
            continue
        seen.add(lab)
        out.append(r)
    return out


def supersede_prompt(new: dict, cands: list[dict], *, day: int) -> list[dict]:
    lines = [f"{n + 1}. 第{c.get('day', 0)}天｜{row_text(c)}" for n, c in enumerate(cands)]
    user = f"今天是第{day}天。\n新紀錄：第{new.get('day', day)}天｜{row_text(new)}\n舊紀錄：\n" + "\n".join(lines)
    return [{"role": "system", "content": SUPERSEDE_SYS}, {"role": "user", "content": user}]


async def supersede(ask, new_records: list[dict], older_records: list[dict], index, *, day: int | None = None,
                    k: int = 3) -> list[tuple]:
    """[(old_id, new_id, reason)] for today's records (`new_records`) against `older_records` (not yet superseded),
    using the index to pick candidates. All records need "id"."""
    by_id = {r["id"]: r for r in older_records}
    done: set = set()
    marks = []
    for new in new_records:
        if not clean(new["value"])[1]:
            continue                              # a listening row that is not a fact supersedes nothing
        if not value_of_kind(new["aspect"], new["value"]):
            continue                              # 備用鑰匙｜放在哪｜拿去: not a place, replaces nothing
        avail = [i for i in by_id if i not in done]
        if not avail:
            break
        ranked = index.scores(key_text(new), avail)[:k]
        cands = [by_id[i] for i, _ in ranked
                 if by_id[i]["aspect"] == new["aspect"] and same_thing(by_id[i]["thing"], new["thing"])]
        if not cands:
            continue
        d = day if day is not None else new.get("day", 0)
        first = ((await ask(supersede_prompt(new, cands, day=d), SUPERSEDE_TOKENS)) or "").strip().split("\n")[0]
        if "無" in first:
            continue
        for m in re.findall(r"\d+", first):
            j = int(m)
            if 1 <= j <= len(cands) and cands[j - 1]["id"] not in done:
                old = cands[j - 1]
                done.add(old["id"])
                marks.append((old["id"], new["id"], f"{new['thing']}｜{ASPECT_ZH.get(new['aspect'], new['aspect'])}："
                                                    f"{old['value']} → {new['value']}"))
    return marks
