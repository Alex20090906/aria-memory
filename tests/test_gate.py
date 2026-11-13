"""aria gate: energy pool, the four fuses, refractory, idle work, proactive cap, persistence; reflex GRU parity."""
import asyncio
import json
from pathlib import Path

import numpy as np

from aria_memory import Gate, IdleWork, ReflexGRU

FIX = Path(__file__).parent / "fixtures"


def quiet(t, **kw):
    return dict({"t": t, "since_owner_s": 1e9, "has_work": True}, **kw)


def step(content="", units=(), ntok=10, useful=False, thing="傘", aspect="location", agenda=()):
    return {"op": "check", "thing": thing, "aspect": aspect, "content": content, "units": set(units), "ntok": ntok,
            "useful": useful, "agenda": list(agenda)}


def test_energy_comes_only_from_outside():
    g = Gate()
    assert g.tick(quiet(0))["action"] == "rest"                     # no energy at start
    g.owner_spoke(100, 20, day=1)
    assert g.energy == 3
    assert g.tick({"t": 150, "has_work": True})["why"] == "owner spoke recently"
    assert g.tick({"t": 221, "has_work": True})["action"] == "idle_work"
    assert g.tick({"t": 10**7, "has_work": True})["action"] == "idle_work"     # time alone never refills or drains
    g.energy = 9.5
    g.owner_spoke(10**7, 5, day=2)
    assert g.energy == 10                                            # cap


def test_busy_and_no_work_mean_rest():
    g = Gate()
    g.energy = 5
    for kw in ({"in_turn": True}, {"playing": True}, {"vad": True}, {"has_work": False}):
        assert g.tick(quiet(1000, **kw))["action"] == "rest"


def test_surprise_adds_one_at_most_every_ten_minutes_and_fatigue_rests():
    g = Gate(f_rest=1.5)
    g.tick(quiet(0, level="high"))
    g.tick(quiet(300, level="high"))
    assert g.energy == 1
    g.tick(quiet(600, level="high"))
    assert g.energy == 2
    g.tick(quiet(700, level="calibrating"))
    assert g.energy == 2
    assert g.tick(quiet(800, F=2.0))["why"] == "fatigue"


def test_mic_fault_gives_no_energy():
    g = Gate()
    g.tick(quiet(0, faults={"mic": True}))
    g.owner_spoke(10, 5, day=1)
    assert g.energy == 0


def test_fuse_smr():
    g = Gate()
    g.energy = 10
    g.last_ext_t = 0
    assert g.step_done(10, step("a b c d", {1}))["fuse"] is None
    r = g.step_done(20, step("e f g h", {2}))
    assert r["fuse"] == "SMR=0x2" and g.energy == 0


def test_fuse_rep3_sets_refractory():
    g = Gate()
    g.energy = 10
    g.last_ext_t = 0
    g.step_done(10, step("提醒使用者週三看牙醫", {1}, useful=True))
    r = g.step_done(20, step("提醒使用者週三看牙醫喔", {2}, useful=True, thing="牙醫", aspect="time"))
    assert r["fuse"] == "rep3>0.30"
    assert not g.refractory_ok("牙醫", "time", 21) and g.refractory_ok("牙醫", "time", 20 + 1801)


def test_fuse_ovl_needs_two_in_a_row():
    g = Gate()
    g.energy = 10
    g.last_ext_t = 0
    g.step_done(10, step("甲", {1, 2, 3}, useful=True))
    assert g.step_done(20, step("乙", {1, 2, 3}, useful=True))["fuse"] is None
    assert g.step_done(30, step("丙", {1, 2, 3}, useful=True))["fuse"] == "Ovl>0.65x2"


def test_fuse_etr_needs_quiet_outside():
    g = Gate()
    g.energy = 10
    g.owner_spoke(0, 10, day=1)
    r = g.step_done(100, step("一二三四", {1}, ntok=100, useful=True))
    assert r["etr"] > 0.8 and r["fuse"] is None                     # outside input 100 s ago
    r = g.step_done(200, step("五六七八", {9}, ntok=100, useful=True))
    assert r["fuse"] == "ETR>0.8"


def test_agenda_cap_replace_and_morning_speech():
    g = Gate()
    for i in range(7):
        g.add_agenda({"thing": f"事{i}", "aspect": "time", "text": f"提醒{i}", "made": 100 + i})
    g.add_agenda({"thing": "事6", "aspect": "time", "text": "提醒6改", "made": 200})
    assert len(g.agenda) == 5 and g.agenda[-1]["text"] == "提醒6改"
    g.owner_spoke(50_000, 5, day=2)                                 # first hearing of the day
    out = g.tick({"t": 50_001})
    assert out["action"] == "speak" and len(out["items"]) == 3
    assert g.tick({"t": 50_002})["action"] != "speak"
    g.owner_spoke(50_100, 5, day=2)
    assert g.morning_pending is None                                 # not a new morning
    g.add_agenda({"thing": "新", "aspect": "time", "text": "新提醒", "made": 60_000})
    g.owner_spoke(60_000 + 5 * 3600, 5, day=3)
    items = g.tick({"t": 60_000 + 5 * 3600 + 1})["items"]
    assert len(items) == 3 and all(x["made"] < 60_000 + 5 * 3600 for x in items)


def test_state_roundtrip():
    g = Gate(f_rest=2.0)
    g.owner_spoke(10, 30, day=1)
    g.step_done(200, step("內容", {1, 2}, useful=True))
    g.add_agenda({"thing": "a", "aspect": "time", "text": "x", "made": 1})
    g2 = Gate.from_state(json.loads(json.dumps(g.to_state())))
    assert g2.to_state() == g.to_state()


def test_idle_work_order_and_steps():
    w = IdleWork()
    w.note_record({"thing": "傘", "aspect": "location"})
    w.note_record({"thing": "牙醫", "aspect": "time"})
    assert w.next_item(0)[:3] == ("plan", "牙醫", "time")
    w.note_unanswered("機車鑰匙", "location", "我機車鑰匙咧")
    assert w.next_item(0)[0] == "ask"

    async def ask(messages, max_tokens):
        assert max_tokens == 48 and "記憶：[牙醫|time] -> 第3天下午2點半" in messages[1]["content"]
        return "提醒 | 明天下午兩點半要看牙醫"

    vals = {("牙醫", "time"): "第3天下午2點半", ("傘", "location"): "門口"}
    run = lambda item: asyncio.run(w.run(ask, item, latest_value=lambda e, s: vals.get((e, s)),
                                         units_of=lambda s: {len(s)}, day=2, t=5.0))
    s1 = run(w.next_item(0))
    assert s1["op"] == "ask" and s1["useful"] and "機車鑰匙" in s1["agenda"][0]["text"]
    w.commit(s1)
    s2 = run(w.next_item(0))
    assert s2["op"] == "plan" and s2["agenda"][0]["text"] == "明天下午兩點半要看牙醫" and s2["useful"]
    w.commit(s2)
    s3 = run(w.next_item(0))
    assert s3["op"] == "check" and s3["thing"] == "傘" and s3["content"] == "門口" and not s3["useful"]
    w.commit(s3)
    w.commit(run(w.next_item(0)))
    assert w.next_item(0) is None
    assert IdleWork.from_state(json.loads(json.dumps(w.to_state()))).to_state() == w.to_state()
    w.new_day()
    assert not w.has_work(0)


def test_reflex_gru_matches_the_research_model():
    m = ReflexGRU.load(FIX / "reflex_gru_w6r.npz")
    assert m.n_params() == 231_686 and m.d_in == 1029 and m.hidden == 128
    g = np.load(FIX / "reflex_gru_golden.npz")
    h = None
    for x, ref in zip(g["x"], g["out"]):
        out, h = m.step(x, h)
        assert np.allclose(out, ref, atol=1e-4)
    assert np.allclose(h, g["h"], atol=1e-4)
    action, encode, _ = m.decide(g["x"][1])
    assert action in ("WAIT", "ATTEND", "REPLY", "THINK", "INTERRUPT") and isinstance(encode, bool)
