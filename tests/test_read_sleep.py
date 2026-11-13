"""aria reading (render, retrieve-then-read, sound-twin second read) and the night pass (relisten, H42b supersede)."""
import asyncio
from pathlib import Path

from aria_memory import NetIndex, READ_SYS, Vocab, read, relisten, render_records, sound_twins, supersede
from aria_memory.sleep import parse_relisten
from conftest import small_vocab

FIX = Path(__file__).parent / "fixtures"


class Ask:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def __call__(self, messages, max_tokens):
        self.calls.append((messages, max_tokens))
        return self.replies.pop(0) if self.replies else "不知道\n我不記得了。"


def run(coro):
    return asyncio.run(coro)


def rec(i, thing, aspect, value, heard, day, src="owner"):
    return {"id": i, "thing": thing, "aspect": aspect, "value": value, "heard": heard, "day": day, "src": src}


TOP = [rec(7, "護照", "location", "黑色後背包夾層", "護照我改放黑色後背包夾層了", 3),
       rec(2, "護照", "location", "書房防潮箱", "我護照收進書房防潮箱了", 1),
       rec(5, "嫁接刀", "location", "工具箱", "嫁接刀我放工具箱了", 2, src="uncertain")]


def test_render_lists_oldest_first_and_marks_uncertain_speakers():
    text = render_records(TOP, now_day=4)
    lines = text.splitlines()
    assert lines[0] == "1. 第1天｜護照｜放在哪｜書房防潮箱（原話：我護照收進書房防潮箱了）"
    assert lines[1].startswith("2. 第2天｜嫁接刀｜放在哪｜工具箱（原話：嫁接刀我放工具箱了；說話的人不確定")
    assert lines[2].startswith("3. 第3天｜護照")
    assert "（原話：(sleep)）" in render_records([rec(1, "傘", "location", "門口", "", 1, "sleep")])


def test_read_picks_the_numbered_record():
    ask = Ask("編號 3\n護照在黑色後背包夾層。")
    out = run(read(ask, "Aria我護照咧", TOP, now_day=4))
    assert out["status"] == "answer" and out["rec_id"] == 7 and out["quote"] == "護照我改放黑色後背包夾層了"
    assert out["reply"] == "護照在黑色後背包夾層。" and out["mode"] == "read"
    msgs, n = ask.calls[0]
    assert n == 80 and msgs[0]["content"] == READ_SYS
    assert msgs[1]["content"].startswith("今天是第4天。\n使用者問：Aria我護照咧\n記憶紀錄：\n1. 第1天｜護照")


def test_read_unknown_and_empty():
    assert run(read(Ask("不知道\n我不記得了。"), "我機車鑰匙放哪", TOP, now_day=4))["status"] == "unknown"
    assert run(read(Ask(), "什麼", [], now_day=1))["status"] == "unknown"
    assert run(read(Ask("編號 9\n亂講"), "我護照咧", TOP, now_day=4))["status"] == "unknown"


def test_sound_twin_second_read():
    assert sound_twins("我架接刀放哪", TOP) == [(TOP[2], "架接刀")]
    ask = Ask("不知道\n不記得。", "編號 1\n在工具箱。")
    out = run(read(ask, "我架接刀放哪", TOP, now_day=4))
    assert out["status"] == "answer" and out["rec_id"] == 5 and out["mode"] == "read_phon"
    assert "唸起來一樣" in ask.calls[1][0][1]["content"]


def test_parse_relisten_rows():
    said = ["我約後天下午三點剪頭髮", "熱水壺壞了週末要拿去修", "Aria你好"]
    text = "剪頭髮｜時間｜後天下午三點｜1\n熱水壺｜狀態｜壞了，週末要拿去修｜2\n使用者｜方面｜亂｜9\n東西｜顏色｜紅\nNONE"
    rows = parse_relisten(text, said, day=4)
    assert rows[0] == {"thing": "剪頭髮", "aspect": "time", "value": "第6天下午三點", "day": 4,
                       "heard": "我約後天下午三點剪頭髮", "src": "sleep"}
    assert rows[1]["aspect"] == "status" and rows[1]["heard"] == said[1]
    assert len(rows) == 2


def test_relisten_skips_known_labels():
    ask = Ask("剪頭髮｜時間｜後天下午三點｜1\n熱水壺｜狀態｜壞了｜2")
    rows = run(relisten(ask, ["我約後天下午三點剪頭髮", "熱水壺壞了"], day=4, known_labels=[("熱水壺", "status")]))
    assert [r["thing"] for r in rows] == ["剪頭髮"]
    assert ask.calls[0][1] == 600 and ask.calls[0][0][1]["content"].startswith("今天是第4天（星期四）。\n今天使用者說過的話：\n1. ")
    assert run(relisten(Ask(), [], day=4)) == []


def test_supersede_strict():
    vocab = small_vocab()
    older = [rec(0, "護照", "location", "書房防潮箱", "我護照收進書房防潮箱了", 1),
             rec(1, "護照", "status", "快過期了", "護照快過期了", 1),
             rec(2, "咖啡", "preference", "熱美式", "咖啡我都喝熱美式", 1)]
    new = [rec(3, "護照", "location", "黑色後背包夾層", "護照我改放黑色後背包夾層了", 3),
           rec(4, "鑰匙", "location", "不知道掛在哪裡（使用者正在尋找）", "鑰匙咧", 3),
           rec(5, "咖啡", "preference", "冰燕麥拿鐵", "咖啡我改喝冰燕麥拿鐵", 3)]
    ix = NetIndex(vocab)
    for r in older + new:
        ix.add(r["id"], r)
    ask = Ask("1", "無")
    marks = run(supersede(ask, new, older, ix, day=3))
    assert marks == [(0, 3, "護照｜放在哪：書房防潮箱 → 黑色後背包夾層")]
    first_user = ask.calls[0][0][1]["content"]
    assert "新紀錄：第3天｜護照｜放在哪｜黑色後背包夾層" in first_user and "快過期了" not in first_user   # same aspect only
    assert len(ask.calls) == 2                       # the non-fact row (鑰匙) asked nothing


def test_supersede_needs_the_same_thing_and_a_value_of_the_aspects_kind():
    """P4 run with the LLM server, set_old: 牙醫 replaced by 洗牙, 咖啡 by 飲料, and a location 拿去. None of them asks the LLM now."""
    vocab = small_vocab()
    older = [rec(0, "牙醫", "time", "第3天下午2:30", "牙醫約週三下午兩點半", 1),
             rec(1, "咖啡", "preference", "冰燕麥拿鐵不加糖", "咖啡我喝冰燕麥拿鐵不加糖", 1),
             rec(2, "備用鑰匙", "location", "鞋櫃上方小盒子", "備用鑰匙放鞋櫃上方小盒子", 1),
             rec(3, "瑜伽課", "time", "第4天晚上7:30", "瑜伽課週四晚上七點半", 1)]
    new = [rec(4, "洗牙", "time", "第10天下午3點", "洗牙改到下週三下午三點", 3),
           rec(5, "飲料", "preference", "烏龍拿鐵無糖去冰", "飲料幫我點烏龍拿鐵無糖去冰", 3),
           rec(6, "備用鑰匙", "location", "拿去", "備用鑰匙你拿去", 3),
           rec(7, "瑜伽課", "time", "改成別的老師", "瑜伽課改成別的老師", 3)]
    ix = NetIndex(vocab)
    for r in older + new:
        ix.add(r["id"], r)
    ask = Ask("1", "1", "1", "1")
    assert run(supersede(ask, new, older, ix, day=3)) == [] and ask.calls == []


def test_same_thing_and_value_kinds():
    from aria_memory.sleep import same_thing, value_of_kind
    assert same_thing("護照", "護照") and same_thing("我的護照", "護照的位置") and same_thing("ＨＰ墨水", "HP 墨水")
    assert not same_thing("牙醫", "洗牙") and not same_thing("咖啡", "飲料") and not same_thing("印表機", "印表機墨水")
    for v in ("黑色後背包夾層", "公司辦公桌抽屜", "玄關吸鐵盒", "托特包", "更衣室", "常穿外套左邊口袋", "陽台"):
        assert value_of_kind("location", v), v
    for v in ("拿去", "借給小傑", "不見了"):
        assert not value_of_kind("location", v), v
    for v in ("第10天下午3點", "下週（第11天起）", "第15天", "第23天下午2:30", "明天早上"):
        assert value_of_kind("time", v), v
    for v in ("改成別的老師", "取消了"):
        assert not value_of_kind("time", v), v
    assert value_of_kind("status", "壞了")  # other aspects are not checked
