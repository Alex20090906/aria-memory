"""Deterministic rules around the listener (no model calls).

    resolve / fix_reldate   H29: relative dates by rule, not by the model (29/29 against the research scripts)
    clean                   H40: drop listening values that are questions or the listener's own commentary
    strip_suffix            entity names without 我的/那個 and slot words (偏好, 預約, 位置 ...)

Days are counted from 1. The research scripts started on a Monday (day 1 = Monday); in the product pass the real
weekday of `day` (1 = Monday .. 7 = Sunday).
"""
from __future__ import annotations

import re

WD = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
      "6": 6, "7": 7}
CN = {"一": 1, "兩": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
WEEK = r"(?:週|周|禮拜|星期)"
PAT = re.compile(rf"(大後天|後天|明天|今天|昨天|前天|(?P<n>[一兩二三四五六七八九十\d]+)天後|"
                 rf"(?P<pre>下下|下個|下|這個|這|本)?{WEEK}(?P<wd>[一二三四五六日天1-7]))")
DAYN = re.compile(r"第\s*\d+\s*天")
SIMPLE = {"今天": 0, "明天": 1, "後天": 2, "大後天": 3, "昨天": -1, "前天": -2}


def weekday_of(day: int, weekday: int | None = None) -> int:
    """1 = Monday. Without a real weekday the research convention applies (day 1 = Monday)."""
    return weekday if weekday else (day - 1) % 7 + 1


def resolve(text: str, day: int, weekday: int | None = None) -> list[int]:
    """Day numbers meant by the relative date expressions in `text`. A bare 週X means the next such day (today
    included). Speech recognition errors are not guessed at."""
    out = []
    monday = day - weekday_of(day, weekday) + 1
    for m in PAT.finditer(text or ""):
        g = m.group(0)
        if g in SIMPLE:
            out.append(day + SIMPLE[g])
            continue
        if m.group("n"):
            n = m.group("n")
            n = int(n) if n.isdigit() else CN.get(n)
            if n:
                out.append(day + n)
            continue
        wd = WD[m.group("wd")]
        pre = m.group("pre") or ""
        if pre in ("下", "下個"):
            out.append(monday + 7 + wd - 1)
        elif pre == "下下":
            out.append(monday + 14 + wd - 1)
        elif pre in ("這", "這個", "本"):
            out.append(monday + wd - 1)
        else:
            d = monday + wd - 1
            out.append(d if d >= day else d + 7)
    return out


def fix_reldate(aspect: str, value: str, heard: str, day: int, weekday: int | None = None) -> str:
    """Time/plan values only, and only when the heard sentence names exactly one day: the day in the value is set by
    the rule."""
    if aspect not in ("time", "plan"):
        return value
    days = sorted(set(resolve(heard, day, weekday)))
    if len(days) != 1:
        return value
    if DAYN.search(value):
        return DAYN.sub(f"第{days[0]}天", value, count=1)
    m = PAT.search(value)
    if m:
        return value[:m.start()] + f"第{days[0]}天" + value[m.end():]
    return value


def resolve_value(value: str, day: int, weekday: int | None = None) -> str:
    """Night rows keep relative times as said (H30); the rule turns a single relative date into 第N天."""
    days = sorted(set(resolve(value, day, weekday)))
    m = PAT.search(value)
    if m and len(days) == 1:
        return value[:m.start()] + f"第{days[0]}天" + value[m.end():]
    return value


META = re.compile(r"[（(][^（）()]*?(註|推測|語音辨識|詢問|使用者|略過|不列|語意|同上|未回答|未提供)[^（）()]*?[）)]")
NOT_FACT = re.compile(r"^(未知|位置不明|不明|詢問|使用者詢問|不知道|不清楚)|忘記了|正在尋找|尋找中")
SAYS_NOT_FACT = re.compile(r"非陳述|不列|無答案|未回答|略過")


def clean(value) -> tuple[str, bool]:
    """(cleaned value, keep?). Comments about the listening are trimmed; a value that is not a fact is dropped."""
    v = str(value)
    comments = "".join(m.group(0) for m in META.finditer(v))
    c = META.sub("", v).strip(" ，,。")
    return c, bool(c) and not NOT_FACT.search(c) and not SAYS_NOT_FACT.search(comments)


SUFFIX = ["的偏好", "偏好", "的預約", "預約", "的位置", "位置", "的時間", "時間", "的狀態", "狀態", "的計畫", "計畫",
          "的價格", "價格", "放哪", "在哪"]


def strip_suffix(e: str) -> str:
    e = re.sub(r"^(我的|我|那個|這個)", "", e.strip())
    for s in SUFFIX:
        if e.endswith(s) and len(e) > len(s):
            return e[: -len(s)]
    return e


# clock times (used by the corrected replay score and by what the always-on gate may say)
TIME = re.compile(r"(早上|上午|凌晨|半夜|中午|下午|傍晚|晚上)?\s*(\d{1,2})\s*(?:點|:|：)\s*(半|\d{1,2})?")
AM, PM = {"早上", "上午", "凌晨", "半夜"}, {"下午", "傍晚", "晚上"}


def clock_times(text: str) -> list[tuple[int | None, int, int]]:
    """Clock times in a text: (hour 0-23 when the part of day is said, else None), hour 1-12, minute."""
    from .eval import norm
    out = []
    for part, h, m in TIME.findall(norm(text)):
        h = int(h)
        minute = 30 if m == "半" else int(m or 0)
        if part in PM and h < 12:
            full = h + 12
        elif part == "中午":
            full = h + 12 if h < 6 else h
        elif part in AM:
            full = h % 12
        else:
            full = h if h > 12 else None
        out.append((full, h % 12, minute))
    return out
