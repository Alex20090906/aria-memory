"""aria listening: parser, staged commit, relative dates, false-fact cleaning, the assistant's name in the prompts."""
import asyncio

from aria_memory import LISTEN_SYS, READ_SYS, Listener, clean, listen, prompts, resolve
from aria_memory.listen import parse
from aria_memory.rules import fix_reldate, strip_suffix


class Ask:
    """Scripted LLM: returns the given replies in order and records the prompts."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def __call__(self, messages, max_tokens):
        self.calls.append((messages, max_tokens))
        return self.replies.pop(0) if self.replies else "NONE"


def run(coro):
    return asyncio.run(coro)


def test_prompts_use_the_assistant_name():
    assert prompts.NAME == "Aria"                      # the research scripts' name, also the default wake word
    assert prompts.NAME in LISTEN_SYS and prompts.NAME in READ_SYS
    assert any(prompts.NAME in a for a, _ in prompts.LISTEN_SHOTS)


def test_parse_facts_questions_topic_and_stage_ops():
    ops = []
    facts, qs, topic = parse(["TOPIC 護照", "我的護照 | location | 書房防潮箱", "Q 牙醫預約 | time", "Q 教練 | name",
                              "改口 #1 | 黑色後背包", "垃圾 | color | 紅", "NONE"], ops)
    assert facts == [("護照", "location", "書房防潮箱")]          # unknown slot dropped, 我的 stripped
    assert qs == [("牙醫", "time"), ("教練", "name")]            # any aspect routed (H32)
    assert topic == "護照"
    assert ops == [("改口", 1, "黑色後背包")]


def test_staged_commit_keeps_a_fact_two_sentences_and_applies_a_correction():
    lis = Listener()
    ask = Ask("TOPIC 護照\n護照 | location | 書房防潮箱",
              "TOPIC 護照\n改口 #1 | 黑色後背包夾層",
              "TOPIC 咖啡\n咖啡 | preference | 冰燕麥拿鐵",
              "NONE")
    r1 = run(lis.hear(ask, "我護照收進書房防潮箱了", day=1))
    assert r1.records == [] and lis.staged()[0]["value"] == "書房防潮箱"
    r2 = run(lis.hear(ask, "啊不對，是黑色後背包夾層", day=1))
    assert r2.records == [] and lis.staged()[0]["value"] == "黑色後背包夾層"
    r3 = run(lis.hear(ask, "咖啡我現在喝冰燕麥拿鐵", day=1))
    assert r3.records == []
    r4 = run(lis.hear(ask, "好熱喔", day=1))
    assert [(r["thing"], r["value"], r["heard"]) for r in r4.records] == [("護照", "黑色後背包夾層", "我護照收進書房防潮箱了")]
    rest = lis.flush()
    assert [(r["thing"], r["value"]) for r in rest] == [("咖啡", "冰燕麥拿鐵")]
    assert lis.topic == "（無）" and lis.staged() == []


def test_listener_prompt_shape_and_state_roundtrip():
    lis = Listener()
    ask = Ask("TOPIC 充電線\n充電線 | location | 電視櫃抽屜")
    run(lis.hear(ask, "就放電視櫃抽屜啦", day=3, recent=[("欸Aria那個充電線我收起來了", True), ("新聞說颱風", False)]))
    msgs, max_tokens = ask.calls[0]
    assert max_tokens == 160
    assert msgs[0]["content"].startswith(LISTEN_SYS) and "暫存" in msgs[0]["content"]
    assert len(msgs) == 1 + 2 * len(prompts.LISTEN_SHOTS) + 1
    user = msgs[-1]["content"]
    assert user.startswith("今天是第3天（星期三）。\nTOPIC：（無）")
    assert "[使用者] 欸Aria那個充電線我收起來了\n[其他人] 新聞說颱風" in user
    assert user.endswith("剛聽到的一句：\n[使用者] 就放電視櫃抽屜啦")
    lis2 = Listener.from_state(lis.state())
    assert lis2.state() == lis.state() and lis2.topic == "充電線"


def test_relative_dates_by_rule_with_real_weekday():
    assert resolve("下禮拜一出貨", 12) == [15]
    assert resolve("牙醫改到下禮拜二早上十點", 5) == [9]
    assert resolve("禮拜五前要出貨", 8) == [12]
    assert resolve("明天早上十點", 3) == [4]
    # the product passes the real weekday: day 10 is a Sunday here, so 下禮拜二 is day 12
    assert resolve("下禮拜二", 10, weekday=7) == [12]
    assert fix_reldate("time", "第8天早上10點", "牙醫改到下禮拜二早上十點", 5) == "第9天早上10點"
    assert fix_reldate("location", "第8天", "明天", 5) == "第8天"


def test_listen_fixes_the_date_and_cleans_non_facts():
    ask = Ask("TOPIC 牙醫\n牙醫 | time | 第8天早上10點\n鑰匙 | location | 不知道掛在哪裡（使用者正在尋找）")
    recs = run(listen(ask, "牙醫改到下禮拜二早上十點", day=5))
    assert recs == [{"thing": "牙醫", "aspect": "time", "value": "第9天早上10點", "day": 5,
                     "heard": "牙醫改到下禮拜二早上十點", "src": "owner"}]


def test_clean_rules():
    assert clean("詢問在哪（同上，無答案，略過）")[1] is False
    assert clean("濾網已清洗，正在陽台晾乾（註：此句未涉及#1）") == ("濾網已清洗，正在陽台晾乾", True)
    assert clean("書房防潮箱") == ("書房防潮箱", True)
    assert strip_suffix("我的咖啡偏好") == "咖啡"
