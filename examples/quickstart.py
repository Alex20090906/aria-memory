"""Quickstart: no GPU, no model download, about 10 seconds on a laptop CPU.

    pip install -e .
    python examples/quickstart.py

1. Builds a memory index over a few spoken records and asks it questions (CharVocab: no model at all).
2. Re-scores the logged answers of the final research system (E49) with the strict rules: 30 right, 6 wrong,
   20 of 21 never-told questions held.
3. Runs the index offline on both 21-day scripts: is a right record in the top 5?
4. Shows how much of the scripts a wake-word-only microphone would still hear.

Every number printed here comes from files in tests/fixtures/ (synthetic speech, see docs/REVIEW.md).
To use the thinker model's vocabulary instead of CharVocab, build it (docs/VOCAB.md) and set ARIA_VOCAB to its folder.
"""
import json
import os
from pathlib import Path

from aria_memory import CharVocab, NetIndex, Vocab, render_records
from aria_memory.eval import ScriptSet, score
from aria_memory.replay import coverage, index_recall

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def rec(i, thing, aspect, value, heard, day):
    return {"id": i, "thing": thing, "aspect": aspect, "value": value, "heard": heard, "day": day, "src": "owner"}


def part1(vocab):
    print("1. A small index (records are never edited; a newer record just ranks with the older one)")
    records = [
        rec(0, "護照", "location", "書房防潮箱", "我護照收進書房防潮箱了", 1),
        rec(1, "太陽眼鏡", "location", "玄關鞋櫃最上面", "太陽眼鏡改放玄關鞋櫃最上面", 1),
        rec(2, "牙醫", "time", "第3天下午2點半", "我約後天下午兩點半看牙醫", 1),
        rec(3, "咖啡", "preference", "冰燕麥拿鐵", "我現在喝咖啡都改冰燕麥拿鐵", 2),
        rec(4, "護照", "location", "黑色後背包夾層", "護照我改放黑色後背包夾層了", 3),
    ]
    ix = NetIndex(vocab)
    for r in records:
        ix.add(r["id"], r)
    by_id = {r["id"]: r for r in records}
    for q in ("Aria，我護照咧？", "我太陽眼鏡放哪", "牙醫是幾點"):
        top = ix.query(q, k=3)
        print(f"   Q: {q}")
        print("      top 3:", ", ".join(f"#{i} {by_id[i]['thing']}={by_id[i]['value']} ({s:.2f})" for i, s in top))
    print("   What the reader (any LLM) would be given for the first question:")
    top = [by_id[i] for i, _ in ix.query("Aria，我護照咧？", k=5)]
    for line in render_records(top, now_day=4).splitlines():
        print("      " + line)
    print("   The reader is told: use a record only if it is about the same thing and aspect; newest day wins.")


def part2():
    print("\n2. Strict re-scoring of E49's logged answers (tests/fixtures/alwayson/e49_answers.json)")
    old = ScriptSet(FIX / "set_old")
    e49 = json.loads((FIX / "alwayson" / "e49_answers.json").read_text(encoding="utf-8"))
    r = score(e49["answers"], old)
    print(f"   told questions: right {r['told_right']}, wrong {r['told_wrong']}, of {r['told_total']}")
    print(f"   never-told questions held (said 'I don't know'): {r['untold_held']} of {r['untold_total']}")


def part3(vocab, label):
    print(f"\n3. Index recall on the 21-day scripts with {label} (a right record in the top 5?)")
    for name, research in (("set_old", 35), ("set2", 34)):
        r = index_recall(ScriptSet(FIX / name), vocab)
        print(f"   {name}: {r['top5']}/{r['answerable']}   (research, model vocabulary: {research}/38)")


def part4():
    print("\n4. Wake-word coverage of set_old (told questions whose audio would be transcribed)")
    old = ScriptSet(FIX / "set_old")
    for cond, what in (("all", "always listening"), ("wake", "wake word only"), ("wake8", "wake word + 8 s window")):
        c = coverage(old, cond)
        print(f"   {what:24s} questions {c['told_questions_taken']}/{c['told_questions']}, "
              f"facts {c['facts_taken']}/{c['facts']}")


def main():
    folder = os.environ.get("ARIA_VOCAB")
    if folder:
        vocab, label = Vocab.load(folder), "the model vocabulary"
    else:
        vocab, label = CharVocab(), "CharVocab (no model)"
    part1(vocab)
    part2()
    part3(vocab, label)
    part4()


if __name__ == "__main__":
    main()
