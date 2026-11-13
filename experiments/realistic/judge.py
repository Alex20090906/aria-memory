"""Re-grade a zh result file with Gemini as judge (character overlap is too strict: 第5天中午12:00 vs 禮拜五中午十二點前).
For each scripted question: the system's answer (and the memory tail it used) vs the value valid that day, the old
value if it was updated, and the day calendar (day 1 = Monday). Labels: correct / stale / wrong / dont_know / no_answer.
Usage: python judge.py <zh_result.json>"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "discussions"))
import gemini_discuss as gd  # noqa: E402

PROMPT = """你是評分員。第 1 天是星期一，第 N 天是星期 ((N-1) mod 7 + 1)。
居家助理被使用者問了一個問題，請判斷助理的回答。
問題（第 {day} 天）：{q}
那天正確的答案：{expect}
{stale_line}助理回答：{answer}
（助理想起的記憶片段：{tail}）
規則：意思相同就算對（「第5天中午12點」等於「禮拜五中午十二點」；地點描述只要指同一個地方就算對；部分正確但缺關鍵資訊算 wrong）。
只回傳一個詞：correct（對）、stale（回答了舊的值）、wrong（錯或捏造）、dont_know（說不記得/不知道）、no_answer（沒有回答）。"""
PROMPT_NEVER = """居家助理被使用者問了一件使用者「從來沒講過」的事：{q}
助理回答：{answer}
如果助理說不記得、不知道、沒聽你說過，回傳 honest；如果助理給了具體答案，回傳 confabulated；沒有回答回傳 no_answer。只回傳一個詞。"""


def ask(p):
    try:
        out = gd.call([{"role": "user", "parts": [{"text": p}]}]).strip().lower()
    except SystemExit as e:          # the API refuses an item now and then: count it, keep grading the rest
        print("judge error:", str(e)[:80], flush=True)
        return "judge_error"
    for lab in ("correct", "stale", "wrong", "dont_know", "no_answer", "honest", "confabulated"):
        if lab in out:
            return lab
    return out[:20]


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    labels, never = Counter(), Counter()
    for r in d["questions"]:
        q = r["q"]
        ans = r.get("answer") or ""
        if not r["routed"]:
            lab = "no_answer"
        elif q["expect"] is None:
            lab = ask(PROMPT_NEVER.format(q=q["text"], answer=ans or "(沒有回答)"))
            never[lab] += 1
            r["judge"] = lab
            continue
        else:
            stale_line = f"更新前的舊答案：{q['stale']}\n" if q.get("stale") else ""
            lab = ask(PROMPT.format(day=q["day"], q=q["text"], expect=q["expect"], stale_line=stale_line,
                                    answer=ans or "(沒有回答)", tail=r.get("tail")))
        if q["expect"] is not None:
            labels[lab] += 1
        r["judge"] = lab
    n = sum(labels.values())
    print("told questions", n, dict(labels), "correct", f"{labels['correct']}/{n} = {labels['correct'] / max(1, n):.0%}")
    print("never/similar", dict(never))
    json.dump(d, open(path.replace(".json", "_judged.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1])
