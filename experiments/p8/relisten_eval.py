"""H30 offline: the night relisten step of zh_v3, old prompt vs a Chinese row format that also names its source sentence.

All zh_v3 result files (E37, E42) and the always-on simulator show night-relisten rows with English labels, slot names
("plan", "location") or "使用者" as the label; daytime listening almost never does. The old prompt writes the row format in
English ("entity | slot | value"). The new prompt writes it in Chinese (東西｜方面｜內容｜句號), asks to keep relative
times as said (the rule of H29 resolves them), and names the sentence each row comes from, so the row can carry its
original sentence like daytime rows (useful to the retrieve-then-read answerer, H28).

Per variant and script set: rows; bad labels (English and not in the day's sentences, slot name, 使用者/我); script facts
covered with the right label (label overlap >= 0.5 and value overlap >= 0.5, the zh_v3 rule, numbers normalised);
facts whose value appears under any label (what retrieve-then-read can use); facts covered only by bad-label rows
(what dropping those rows would lose); relative-time facts covered; rows with a valid source sentence.

Replays the day's owner sentences from asr.json against the normal LLM server service (8081).

    python relisten_eval.py <asr.json> <zh_scripts.json> [<asr.json> <zh_scripts.json> ...]
env: RL_VARIANTS (default "old,zh"), LLM_API_KEY
"""
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from label_finder_eval import overlap  # noqa: E402
from rescore_norm import norm  # noqa: E402
import reldate  # noqa: E402

URL = "http://<thinker-host>:8081/v1/chat/completions"
SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
ZH_SLOT = {v: k for k, v in SLOT_ZH.items()}
SLOTS = list(SLOT_ZH)
SUFFIX = ["的偏好", "偏好", "的預約", "預約", "的位置", "位置", "的時間", "時間", "的狀態", "狀態", "的計畫", "計畫",
          "的價格", "價格", "放哪", "在哪"]
WD = "一二三四五六日"

# zh_v3 RELISTEN_SYS with CANON off (the setting of E37 base and E42)
OLD_SYS = ("你是居家助理 Aria 的思考核心，現在是睡前，你在重新回想今天使用者說過的話（語音辨識結果，可能有錯字）。"
           "整理出今天使用者生活中值得記住的事的「最終版本」：改口以最後說的為準，代名詞補成具體的東西，相對時間換算成第幾天。"
           "每行一件事：「entity | slot | value」，slot 只能是 " + "、".join(SLOTS) + "。entity 用最短最常用的叫法。"
           "別人的事、電視內容、問題、閒聊不要列。沒有就輸出 NONE。")
# Example words occur in neither script set.
ZH_SYS = ("你是居家助理 Aria 的思考核心，現在是睡前，你在重新回想今天使用者說過的話（語音辨識結果，可能有錯字；每句前面有編號）。"
          "整理出今天使用者生活中值得記住的事的「最終版本」：改口以最後說的為準，代名詞補成具體的東西。"
          "相對時間照原話寫（例如「後天下午三點」「下禮拜二早上」），不要自己換算成日期。\n"
          "每行一件事，格式：「東西｜方面｜內容｜句號」，全部用中文寫（原話裡本來就是英文的名字、型號可以照抄）。\n"
          "- 東西：那件東西或那個人的名字，用最短最常用的叫法；不能寫「使用者」「我」，也不能寫方面的名稱。"
          "使用者自己要做的事，東西就寫那件事（例如「游泳課」「剪頭髮」）。\n"
          "- 方面只能是：" + "、".join(SLOT_ZH.values()) + "。\n"
          "- 句號：這件事根據的是第幾句，只寫一個數字。\n"
          "例：熱水壺｜狀態｜壞了，週末要拿去修｜3\n"
          "別人的事、電視內容、問題、閒聊不要列。沒有就輸出 NONE。")


def chat(system, user, max_tokens):
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "system", "content": system},
                                                                         {"role": "user", "content": user}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}, ensure_ascii=False).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return (json.loads(r.read())["choices"][0]["message"]["content"] or "").strip().splitlines()
        except Exception:
            if i == 3:
                raise
            time.sleep(5)


def strip_suffix(e):
    e = re.sub(r"^(我的|我|那個|這個)", "", e.strip())
    for s in SUFFIX:
        if e.endswith(s) and len(e) > len(s):
            return e[: -len(s)]
    return e


def bad_label(e, said):
    if e in ("使用者", "我", "") or e.lower() in SLOT_ZH or e in ZH_SLOT:
        return True
    return not re.search(r"[一-鿿]", e) and e.lower() not in said.lower()


def fix_date(v, src, day):
    """H29 on a relisten value: a relative expression in the value is resolved by rule; a 第N天 the model wrote is
    recomputed from the source sentence when that sentence names exactly one day."""
    days = sorted(set(reldate.resolve(v, day)))
    m = reldate.PAT.search(v)
    if m and len(days) == 1:
        return v[:m.start()] + f"第{days[0]}天" + v[m.end():]
    if src and re.search(r"第\s*\d+\s*天", v):
        sd = sorted(set(reldate.resolve(src, day)))
        if len(sd) == 1:
            return re.sub(r"第\s*\d+\s*天", f"第{sd[0]}天", v, count=1)
    return v


def rows_for_day(variant, day, said):
    dd = f"今天是第{day}天（星期{WD[(day - 1) % 7]}）。"
    if variant == "old":
        lines = chat(OLD_SYS, dd + "\n今天使用者說過的話：\n" + "\n".join(f"- {b}" for b in said), 400)
        out = []
        for line in lines:
            if line.count("|") == 2:
                e, s, v = [x.strip() for x in line.split("|")]
                if s in SLOTS and e and v:
                    out.append({"e": strip_suffix(e), "s": s, "v": v, "src": None})
        return out
    lines = chat(ZH_SYS, dd + "\n今天使用者說過的話：\n" + "\n".join(f"{i + 1}. {b}" for i, b in enumerate(said)), 600)
    out = []
    for line in lines:
        parts = [x.strip() for x in re.split(r"[｜|]", line)]
        if len(parts) not in (3, 4):
            continue
        e, s, v = parts[:3]
        s = ZH_SLOT.get(s, s if s in SLOTS else None)
        if not (s and e and v):
            continue
        src = None
        if len(parts) == 4:
            m = re.search(r"\d+", parts[3])
            if m and 1 <= int(m.group()) <= len(said):
                src = said[int(m.group()) - 1]
        out.append({"e": strip_suffix(e), "s": s, "v": fix_date(v, src, day) if s in ("time", "plan") else v,
                    "src": src})
    return out


def run(asr_path, sc_path, variants):
    asr = json.load(open(asr_path, encoding="utf-8"))
    facts = json.load(open(sc_path, encoding="utf-8"))["facts"]
    print("==", asr_path, flush=True)
    for var in variants:
        rows = []
        for d in asr:
            said = [h["text"] for h in d["heard"] if h["owner"] == "owner"]
            for r in rows_for_day(var, d["day"], said):
                r["day"] = d["day"]
                r["bad"] = bad_label(r["e"], " ".join(said))
                rows.append(r)
        cov = val = only_bad = rel = rel_n = 0
        for f in facts:
            near = [r for r in rows if f["day"] <= r["day"] <= f["day"] + 1]
            vhit = [r for r in near if overlap(norm(f["value"]), norm(r["v"])) >= 0.5]
            lhit = [r for r in vhit if overlap(f["entity"], r["e"]) >= 0.5]
            cov += bool(lhit)
            val += bool(vhit)
            only_bad += bool(vhit) and all(r["bad"] for r in vhit)
            if f.get("kind") == "relative_time":
                rel_n += 1
                rel += bool(lhit)
        nb = sum(r["bad"] for r in rows)
        print(f"  {var}: rows {len(rows)} | bad labels {nb} | facts with right label {cov}/{len(facts)} | value under any "
              f"label {val}/{len(facts)} | only under bad label {only_bad} | relative time {rel}/{rel_n} | "
              f"with source sentence {sum(r['src'] is not None for r in rows)}", flush=True)
        for r in rows:
            if r["bad"]:
                print(f"     bad day {r['day']}: {r['e']}｜{r['s']}｜{r['v'][:40]}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    variants = os.environ.get("RL_VARIANTS", "old,zh").split(",")
    a = sys.argv[1:]
    for p, s in zip(a[0::2], a[1::2]):
        run(p, s, variants)
