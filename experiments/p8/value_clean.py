"""H40: clean listening values. The listener sometimes writes a question as a fact ("詢問在哪（同上，無答案，略過）",
"不知道掛在哪裡（使用者正在尋找）") or appends its own commentary to a real value ("濾網已清洗，正在陽台晾乾（註：此句未涉及#1…）").
Rule: drop parenthesised comments that talk about the listening itself; then drop the row if what is left is not a fact
(unknown / being asked / forgotten).

    python value_clean.py      (checks every zh_v3 result file)
"""
import glob
import json
import os
import re
import sys

META = re.compile(r"[（(][^（）()]*?(註|推測|語音辨識|詢問|使用者|略過|不列|語意|同上|未回答|未提供)[^（）()]*?[）)]")
NOT_FACT = re.compile(r"^(未知|位置不明|不明|詢問|使用者詢問|不知道|不清楚)|忘記了|正在尋找|尋找中")
SAYS_NOT_FACT = re.compile(r"非陳述|不列|無答案|未回答|略過")      # the listener's own comment says it is not a fact


def clean(v):
    """(cleaned value, keep?)"""
    v = str(v)
    comments = "".join(m.group(0) for m in META.finditer(v))
    c = META.sub("", v).strip(" ，,。")
    return c, bool(c) and not NOT_FACT.search(c) and not SAYS_NOT_FACT.search(comments)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    here = os.path.dirname(os.path.abspath(__file__))
    n = dropped = changed = 0
    for f in glob.glob(os.path.join(here, "..", "llm_fastmem", "results", "zh_result_*.json")):
        for w in json.load(open(f, encoding="utf-8"))["writes"]:
            n += 1
            c, keep = clean(w["fact"][2])
            if not keep:
                dropped += 1
                print("  drop:", w["fact"][0], "|", str(w["fact"][2])[:50])
            elif c != str(w["fact"][2]):
                changed += 1
                print("  trim:", str(w["fact"][2])[:44], "->", c[:30])
    print(f"writes {n}, dropped {dropped}, trimmed {changed}")
