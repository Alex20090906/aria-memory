"""Rescore zh_v3 results with number and weekday spellings normalised before the character-overlap check.

The scoring in zh_v3 compares characters, so "報價800元" vs expected "報價八百塊" or "本週五中午12點" vs "禮拜五中午十二點"
count as wrong although they say the same thing. This rescoring changes only the spelling, never the content:
Chinese numerals -> Arabic digits, 禮拜/星期/週/周 -> 週, 塊 -> 元, full-width digits -> ASCII. Both scores are reported.

    python rescore_norm.py <zh_result_*.json> [...]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from label_finder_eval import overlap  # noqa: E402

UNKNOWN = re.compile(r"(不記得|不知道|沒有印象|沒聽你說|沒講過|沒說過|忘了|不確定|don't|do not)")
DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(s):
    total, cur = 0, 0
    for ch in s:
        if ch in DIG:
            cur = DIG[ch]
        elif ch in UNIT:
            total += (cur or 1) * UNIT[ch]
            cur = 0
    return total + cur


def norm(s):
    s = s or ""
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = re.sub(r"(禮拜|星期|周)", "週", s)
    s = s.replace("塊", "元").replace("本週", "這週").replace("這禮拜", "這週")
    # weekday names keep their character (週一..週日); other Chinese numerals become digits
    s = re.sub(r"週([一二三四五六日天])", lambda m: "週" + "一二三四五六日日"["一二三四五六日天".index(m.group(1))] + "§", s)
    s = re.sub(r"[零〇一二兩三四五六七八九十百千]+", lambda m: str(cn_to_int(m.group(0))), s)
    s = re.sub(r"週(\d)§", lambda m: "週" + "一二三四五六日"[int(m.group(1)) - 1] if 1 <= int(m.group(1)) <= 7 else m.group(0), s)
    return s.replace("§", "")


def rescore(path):
    d = json.load(open(path, encoding="utf-8"))
    told = [r for r in d["questions"] if r["q"]["expect"] is not None]
    hit_strict = sum(bool(r.get("hit")) for r in told)
    hit_norm, ask_ok, ask_wrong, direct_ok = 0, 0, 0, 0
    for r in told:
        a = (r.get("answer") or "") + " " + str(r.get("tail") or "")
        ok = overlap(norm(r["q"]["expect"]), norm(a)) >= 0.6 and not UNKNOWN.search(r.get("answer") or "")
        hit_norm += ok
        if r.get("mode") == "ask":
            ask_ok += ok
            ask_wrong += not ok
        elif ok:
            direct_ok += 1
    s = d["summary"]
    return {"file": os.path.basename(path), "recall_strict": f"{hit_strict}/{len(told)}", "recall_norm": f"{hit_norm}/{len(told)}",
            "direct_ok": direct_ok, "ask_ok": ask_ok, "ask_wrong": ask_wrong, "never_ask": s.get("never_ask"),
            "honest": s.get("honest"), "coverage": s.get("fact_coverage"), "false_writes": s.get("false_writes_from_others")}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert norm("報價八百塊") == "報價800元", norm("報價八百塊")
    assert norm("禮拜五中午十二點") == "週五中午12點", norm("禮拜五中午十二點")
    assert norm("第13天（禮拜六）下午一點半") == "第13天（週六）下午1點半", norm("第13天（禮拜六）下午一點半")
    for p in sys.argv[1:]:
        print(json.dumps(rescore(p), ensure_ascii=False))
