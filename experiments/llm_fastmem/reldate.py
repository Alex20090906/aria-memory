"""H29: resolve relative dates by rule instead of asking the language model to do calendar arithmetic.

Day 1 is a Monday (as in zh_v3: 第3天（星期三）). resolve(text, day) finds the relative date expressions in a heard
sentence and returns the day numbers they mean: 今天/明天/後天/大後天/昨天/前天, N天後, (這/本)週X, 下週X, 下下週X,
with 週 written as 週/周/禮拜/星期 and X as 一..六/日/天 or 1..7. A bare 週X means the next such day from today
(today included). Speech recognition errors are not guessed at.

    python reldate.py <zh_result_*.json> [...]    check every time/plan write whose heard sentence has exactly one
                                                  relative date: does the written 第N天 match the rule's day?
"""
import json
import os
import re
import sys

WD = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
      "6": 6, "7": 7}
CN = {"一": 1, "兩": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
WEEK = r"(?:週|周|禮拜|星期)"
PAT = re.compile(rf"(大後天|後天|明天|今天|昨天|前天|(?P<n>[一兩二三四五六七八九十\d]+)天後|"
                 rf"(?P<pre>下下|下個|下|這個|這|本)?{WEEK}(?P<wd>[一二三四五六日天1-7]))")


def weekday(day):
    return (day - 1) % 7 + 1          # 1 = Monday


def resolve(text, day):
    out = []
    for m in PAT.finditer(text or ""):
        g = m.group(0)
        if g in ("今天", "明天", "後天", "大後天", "昨天", "前天"):
            out.append(day + {"今天": 0, "明天": 1, "後天": 2, "大後天": 3, "昨天": -1, "前天": -2}[g])
            continue
        if m.group("n"):
            n = m.group("n")
            n = int(n) if n.isdigit() else CN.get(n, None)
            if n:
                out.append(day + n)
            continue
        wd = WD[m.group("wd")]
        pre = m.group("pre") or ""
        monday = day - weekday(day) + 1
        if pre in ("下", "下個"):
            out.append(monday + 7 + wd - 1)
        elif pre == "下下":
            out.append(monday + 14 + wd - 1)
        elif pre in ("這", "這個", "本"):
            out.append(monday + wd - 1)
        else:                                # bare 週X: the next such day, today included
            d = monday + wd - 1
            out.append(d if d >= day else d + 7)
    return out


def check(path):
    d = json.load(open(path, encoding="utf-8"))
    n = agree = 0
    bad = []
    for w in d["writes"]:
        e, s, v = w["fact"][0], w["fact"][1], str(w["fact"][2])
        if s not in ("time", "plan"):
            continue
        days = resolve(w.get("heard", ""), w["day"])
        m = re.search(r"第\s*(\d+)\s*天", v)
        if len(set(days)) != 1 or not m:
            continue
        n += 1
        if int(m.group(1)) == days[0]:
            agree += 1
        else:
            bad.append((w["day"], w["heard"][:40], v, days[0]))
    print(os.path.basename(path), f"time/plan writes with one relative date: {n}, model day == rule day: {agree}")
    for b in bad:
        print("   ", b)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert resolve("下禮拜一出貨", 12) == [15], resolve("下禮拜一出貨", 12)
    assert resolve("牙醫改到下禮拜二早上十點", 5) == [9]
    assert resolve("禮拜五前要出貨", 8) == [12]
    assert resolve("明天早上十點", 3) == [4]
    for p in sys.argv[1:]:
        check(p)
