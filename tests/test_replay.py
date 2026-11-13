"""aria evaluation and replays on the research's 21-day scripts (fixtures/aria)."""
import asyncio
import gzip
import json
import os
from pathlib import Path

import pytest

from aria_memory import Gate, Vocab
from conftest import small_vocab
from aria_memory.eval import ScriptSet, norm, rename_wake, right, score
from aria_memory.replay import coverage, index_recall, replay_cached, simulate_always_on, utterances, windows

FIX = Path(__file__).parent / "fixtures"
REAL = Path(os.environ.get("ARIA_VOCAB", Path(__file__).resolve().parents[1] / "data" / "aria_vocab"))


@pytest.fixture(scope="module")
def old():
    return ScriptSet(FIX / "set_old")


@pytest.fixture(scope="module")
def vocab():
    return small_vocab()


def test_strict_rescoring_reproduces_e49(old):
    e49 = json.loads((FIX / "alwayson" / "e49_answers.json").read_text(encoding="utf-8"))
    r = score(e49["answers"], old)
    assert (r["told_right"], r["told_wrong"], r["untold_held"]) == (30, 6, 20)
    assert (r["told_total"], r["untold_total"]) == (51, 21)


def test_scoring_rules():
    assert right("第21天早上九點", "第21天早上9點")
    assert not right("第21天早上九點", "第9天早上9點")             # a different day number is wrong
    assert norm("禮拜五中午十二點") == "週五中午12點"
    assert rename_wake("欸 Aria，我護照咧") == "欸 Aria，我護照咧"
    assert rename_wake("欸 Aria，我護照咧", wake="Luna") == "欸Luna，我護照咧"


def test_script_lines_and_heard(old):
    lines = old.lines()
    assert len([x for x in lines if x["question"]]) == 72
    assert all(x["text"] == x["text_orig"] for x in lines)        # the default wake word is the scripts' Aria
    owner = [x for x in lines if x["speaker"] in ("owner", "phone_owner")]
    assert len(owner) == 164
    heard = old.heard()
    assert {h["voice"] for h in heard} <= {"owner", "other", "unsure"}
    assert any("Aria" in h["text"] for h in heard)


def test_wake_word_windows(old):
    heard = old.heard()
    allw = windows(heard, "all")
    wake = windows(heard, "wake")
    wake8 = windows(heard, "wake8")
    assert len(allw) == len(heard) and len(wake) < len(wake8) < len(allw)
    for n, utt in enumerate(utterances(heard)):
        if any("Aria" in h["text"] for h in utt):
            assert all(any(w["start"] == h["start"] and w["day"] == h["day"] for w in wake) for h in utt)
            break
    with pytest.raises(ValueError):
        windows(heard, "always")


def test_coverage_under_the_wake_word(old):
    c_all, c_wake, c_w8 = (coverage(old, c) for c in ("all", "wake", "wake8"))
    assert c_all["told_questions_taken"] == 50
    # measured 2026-11-09 on set_old: wake only keeps 24 of 51 told questions, the 8 s window 44
    assert c_wake["told_questions_taken"] == 24 and c_w8["told_questions_taken"] == 44
    assert c_wake["facts_taken"] < c_w8["facts_taken"] < c_all["facts_taken"]


def test_index_recall_runs_on_the_small_vocab(old, vocab):
    r = index_recall(old, vocab)
    assert r["answerable"] == 38 and 25 <= r["top5"] <= 38


def test_replay_cached_with_a_reader_that_never_knows(old, vocab):
    async def ask(messages, max_tokens):
        return "不知道\n我不記得了。"

    r = asyncio.run(replay_cached(old, vocab, ask))
    assert r["told_right"] == 0 and r["untold_held"] == 21 and r["records"] == 135
    r8 = asyncio.run(replay_cached(old, vocab, ask, "wake8"))
    assert r8["records"] < 135 and r8["questions_routed"] < r["questions_routed"]


def test_simulate_always_on_two_days(old, vocab):
    with gzip.open(FIX / "alwayson" / "timeline_21d.json.gz", "rt", encoding="utf-8") as fh:
        tl = json.load(fh)
    presence = {p["day"]: p for p in tl["presence"]}

    async def ask(messages, max_tokens):
        if "提醒" in messages[0]["content"] and max_tokens == 48:
            return "提醒 | 記得明天的事"
        return "不知道\n我不記得了。"

    r = asyncio.run(simulate_always_on(tl["events"], presence, old, vocab, ask, gate=Gate(), days=2))
    assert r["idle_steps"] > 0 and r["vegetative_hours"] == 0
    assert sum(r["fuses"].values()) > 0
    assert r["proactive_said"] <= 3 * 2


@pytest.mark.skipif(not (REAL / "token_units.npy").exists(), reason="real model vocabulary not present")
def test_index_recall_real_vocab_both_sets():
    v = Vocab.load(REAL)
    assert index_recall(ScriptSet(FIX / "set_old"), v)["top5"] == 35
    assert index_recall(ScriptSet(FIX / "set2"), v)["top5"] == 34


# ------------------------------------------------------------------ plan P4 / P5 evaluators
from aria_memory.replay import cached_rows, judge_mark, load_alwayson, p5_report, proactive_score, supersede_eval  # noqa: E402


def test_proactive_score_reproduces_e49(old):
    e49 = json.loads((FIX / "alwayson" / "e49_answers.json").read_text(encoding="utf-8"))
    ps = proactive_score(e49["proactive_said"], old)
    assert ps["spoken"] == 56 and ps["hit_rate"] == 0.696
    assert ps["提醒"] == 32 and ps["提醒_正確"] == 15 and ps["待問"] == 24


def _row(rows, thing, value):
    return next(r for r in rows if r["thing"] == thing and r["value"].startswith(value))


def test_judge_mark_on_the_research_marks(old):
    rows = cached_rows(old)
    # H42b offline (2026-11-06): the mis-heard 「放拿去」 replaced the right place of the spare key
    v, why = judge_mark(old, _row(rows, "備用鑰匙", "鞋櫃上方小盒子"), _row(rows, "備用鑰匙", "拿去"), 17)
    assert v == "wrong" and why.startswith("unsupported")
    v, why = judge_mark(old, _row(rows, "牙醫", "第3天下午2:30"), _row(rows, "洗牙", "第10天下午3點"), 8)
    assert v == "wrong" and why.startswith("crosses things")
    v, _ = judge_mark(old, _row(rows, "咖啡", "冰燕麥拿鐵加糖"), _row(rows, "咖啡", "冰燕麥拿鐵不加糖"), 8)
    assert v == "right"


def test_supersede_eval_with_a_judge_that_never_marks(old, vocab):
    calls = {"sup": 0, "read": 0}

    async def ask(messages, max_tokens):
        if "記憶整理員" in messages[0]["content"]:
            calls["sup"] += 1
            return "無"
        calls["read"] += 1
        return "不知道\n我不記得了。"

    r = asyncio.run(supersede_eval(old, vocab, ask))
    assert r["marks"] == 0 and r["wrong_marks"] == 0 and r["score_before"] == r["score_after"]
    assert r["pass"] == {"score_not_lower": True, "wrong_marks<=1": True}
    assert calls["sup"] > 0 and calls["read"] > 0


def test_p5_report_on_a_short_run(old, vocab):
    events, presence = load_alwayson(FIX / "alwayson")
    assert len(events) == 6114 and len(presence) == 21

    async def ask(messages, max_tokens):
        if max_tokens == 48:
            return "提醒 | 記得明天的事"
        return "不知道\n我不記得了。"

    r = asyncio.run(simulate_always_on(events, presence, old, vocab, ask, gate=Gate(), days=3))
    rep = p5_report(r, presence, old)
    assert rep["hours"] == 72 and rep["vegetative_share"] == 0
    assert rep["away_hours"] > 0 and rep["idle_steps_per_away_hour"] is not None
    assert set(rep["pass"]) == {"vegetative<=1%", "false_wakes_away==0", "reminders>=60%"}
    assert rep["false_wakes"] >= rep["false_wakes_away"] >= 0
