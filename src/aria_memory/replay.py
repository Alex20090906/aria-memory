"""Replays on the research's 21-day scripts.

    index_recall(script, vocab)                       H41 offline check: is a right record in the index's top 5?
                                                      (research: 35/38 set_old, 34/38 set2, block-orthogonal codes)
    windows(heard, condition)                         which heard segments get transcribed under "all" | "wake" | "wake8"
    coverage(script, condition)                       how many told facts and questions survive the wake-word rule
    await replay_cached(script, vocab, ask, condition)  end to end with E43's cached listening, the index and the reader
                                                      (research with the LLM server: right 31 wrong 5 never-told answered 1 on
                                                      set_old, 36 / 4 / 1 on set2)
    await simulate_always_on(...)                     the 21-day always-on loop of E49 with the gate (see its docstring)
    load_alwayson(folder)                             (events, presence) of the 21-day always-on timeline
    p5_report(result, presence, script)               plan P5 numbers from a simulate_always_on result
    await supersede_eval(script, vocab, ask)          plan P4: H42b night marks over the cached rows, wrong marks,
                                                      score with and without superseded records

The scripts call the assistant "Aria", which is also the default wake word (prompts.NAME).
"""
from __future__ import annotations

from collections import defaultdict

import gzip
import json
from pathlib import Path

from .prompts import NAME
from .eval import ScriptSet, norm, overlap, rename_wake, score
from .gate import Gate, IdleWork
from .index import NetIndex
from .read import read
from .sleep import supersede

WAKE = NAME
CONDITIONS = ("all", "wake", "wake8")


# ---------------------------------------------------------------------------------------------- cached records
def cached_rows(script: ScriptSet) -> list[dict]:
    """E43's writes as records with ids, in the order they entered memory (night rows at the end of their day)."""
    rows = []
    for i, w in enumerate(script.cache["writes"]):
        rows.append({"id": i, "thing": w["fact"][0], "aspect": w["fact"][1], "value": str(w["fact"][2]),
                     "day": w["day"], "heard": rename_wake(w.get("heard") or ""), "start": w["start"], "end": w["end"],
                     "src": "sleep" if w["src"] == "sleep" else "owner"})
    return sorted(rows, key=lambda r: (r["day"], r["src"] == "sleep", r["start"]))


def cached_questions(script: ScriptSet, told_only: bool = False) -> list[dict]:
    """Scripted questions with the heard text the listener routed (E43 answers overlapping the question)."""
    out = []
    for r in script.cache["questions"]:
        q = r["q"]
        if told_only and q["expect"] is None:
            continue
        seg = script.segment_of(q)
        if seg is None:
            continue
        cand = [a for a in script.cache["answers"] if a["day"] == q["day"]
                and min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        if not cand:
            continue
        a = max(cand, key=lambda a: overlap(q["entity"], (a["anchor"] or "").split("|")[0]))
        out.append({"q": q, "seg": seg, "heard": rename_wake(a.get("heard") or q["text"]), "start": a["start"],
                    "end": a["end"], "thing": (a["anchor"] or "").strip("[]").split("|")[0]})
    return out


def _available(rows_written: list[dict], q: dict) -> list[dict]:
    return [r for r in rows_written
            if r["day"] < q["q"]["day"] or (r["src"] != "sleep" and r["start"] < q["seg"]["start"])]


def _events(rows, qs):
    ev = [((r["day"], 1 if r["src"] == "sleep" else 0, r["start"]), 0, "w", r) for r in rows]
    ev += [((q["q"]["day"], 0, q["seg"]["start"]), 1, "q", q) for q in qs]
    return sorted(ev, key=lambda x: (x[0], x[1]))


def index_recall(script: ScriptSet, vocab, *, dim: int = 512) -> dict:
    rows = cached_rows(script)
    qs = cached_questions(script, told_only=True)
    ix = NetIndex(vocab, dim=dim)
    written, n, hit, misses = [], 0, 0, []
    for _, _, kind, obj in _events(rows, qs):
        if kind == "w":
            ix.add(obj["id"], obj)
            written.append(obj)
            continue
        avail = _available(written, obj)
        match = {r["id"] for r in avail if overlap(norm(obj["q"]["expect"]), norm(r["value"])) >= 0.6}
        if not match:
            continue
        n += 1
        top = {i for i, _ in ix.scores(obj["heard"], [r["id"] for r in avail])[:5]}
        if top & match:
            hit += 1
        else:
            misses.append(obj["heard"])
    return {"answerable": n, "top5": hit, "misses": misses}


# ---------------------------------------------------------------------------------------------- wake word
def utterances(heard: list[dict], gap_s: float = 1.0) -> list[list[dict]]:
    """Heard segments of a day grouped into utterances (a new one after a pause longer than gap_s)."""
    out, by_day = [], defaultdict(list)
    for h in heard:
        by_day[h["day"]].append(h)
    for day in sorted(by_day):
        cur = []
        for h in sorted(by_day[day], key=lambda x: x["start"]):
            if cur and h["start"] - cur[-1]["end"] > gap_s:
                out.append(cur)
                cur = []
            cur.append(h)
        if cur:
            out.append(cur)
    return out


def windows(heard: list[dict], condition: str = "all", *, follow_s: float = 8.0, answer_s: float = 0.0,
            gap_s: float = 1.0, wake: str = WAKE) -> list[dict]:
    """Heard segments transcribed under a condition (each returned with "utt" = utterance number):
        all     every segment (E49)
        wake    only utterances that contain the wake word
        wake8   as wake, plus utterances starting within follow_s after the end of a transcribed utterance
                (+ answer_s, the assistant's reply time, 0 by default: the conservative window)"""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    out = []
    last_end = {}
    for n, utt in enumerate(utterances(heard, gap_s)):
        day = utt[0]["day"]
        has_wake = any(wake in h["text"] for h in utt)
        take = condition == "all" or has_wake
        if not take and condition == "wake8" and day in last_end:
            take = utt[0]["start"] <= last_end[day] + answer_s + follow_s
        if take:
            last_end[day] = utt[-1]["end"]
            out += [dict(h, utt=n) for h in utt]
    return out


def _taken_spans(script: ScriptSet, condition: str, **kw) -> dict:
    spans = defaultdict(list)
    for h in windows(script.heard(), condition, **kw):
        spans[h["day"]].append((h["start"], h["end"]))
    return spans


def _covered(spans, day, start, end) -> bool:
    return any(min(b, end) - max(a, start) > 0.3 for a, b in spans.get(day, []))


def coverage(script: ScriptSet, condition: str, **kw) -> dict:
    """Owner lines, told facts (their evidence line) and questions whose audio is transcribed under the condition."""
    spans = _taken_spans(script, condition, **kw)
    lines = [ln for ln in script.lines() if ln["speaker"] in ("owner", "phone_owner")]
    taken = [ln for ln in lines if _covered(spans, ln["day"], ln["start"], ln["end"])]
    facts = 0
    for f in script.facts:
        ev = f.get("evidence") or ""
        if any(ln["day"] == f["day"] and ev and ev[:8] in ln["text_orig"] for ln in taken):
            facts += 1
    told = [q for q in script.questions if q["expect"] is not None]
    never = [q for q in script.questions if q["expect"] is None]

    def heard_q(q):
        g = script.segment_of(q)
        return bool(g) and _covered(spans, q["day"], g["start"], g["end"])

    return {"condition": condition, "owner_lines": len(lines), "owner_lines_taken": len(taken),
            "facts": len(script.facts), "facts_taken": facts,
            "told_questions": len(told), "told_questions_taken": sum(heard_q(q) for q in told),
            "never_questions": len(never), "never_questions_taken": sum(heard_q(q) for q in never)}


# ---------------------------------------------------------------------------------------------- end to end (cached)
async def replay_cached(script: ScriptSet, vocab, ask, condition: str = "all", **kw) -> dict:
    """E43/H41 end to end: cached listening, index top 5, reader (ask = the LLM). Under "wake"/"wake8" only records
    and questions whose heard audio is transcribed take part (night rows rest on the day's transcribed sentences:
    a night row whose source sentence was not transcribed is dropped)."""
    spans = _taken_spans(script, condition, **kw) if condition != "all" else None
    heard_by_day = defaultdict(set)
    if spans is not None:
        for h in windows(script.heard(), condition, **kw):
            heard_by_day[h["day"]].add(h["text"])

    def kept_row(r):
        if spans is None:
            return True
        if r["src"] == "sleep":
            return bool(r["heard"]) and r["heard"] in heard_by_day[r["day"]]
        return _covered(spans, r["day"], r["start"], r["end"])

    rows = [r for r in cached_rows(script) if kept_row(r)]
    qs = [q for q in cached_questions(script) if spans is None or _covered(spans, q["q"]["day"], q["start"], q["end"])]
    ix = NetIndex(vocab)
    written, answers = [], []
    for _, _, kind, obj in _events(rows, qs):
        if kind == "w":
            ix.add(obj["id"], obj)
            written.append(obj)
            continue
        avail = _available(written, obj)
        by_id = {r["id"]: r for r in avail}
        top = [by_id[i] for i, _ in ix.scores(obj["heard"], list(by_id))[:5]]
        out = await read(ask, obj["heard"], top, now_day=obj["q"]["day"])
        ok = out["status"] == "answer"
        answers.append({"day": obj["q"]["day"], "start": obj["start"], "end": obj["end"], "thing": obj["thing"],
                        "passed": ok, "value": (out["record"]["value"] + " " + out["reply"]) if ok else "",
                        "reply": out["reply"], "mode": out["mode"]})
    res = score(answers, script)
    res.update({"condition": condition, "records": len(rows), "questions_routed": len(qs), "answers": answers})
    return res


# ---------------------------------------------------------------------------------------------- always-on loop
async def simulate_always_on(events: list[dict], presence: dict, script: ScriptSet, vocab, ask, *,
                             gate: Gate | None = None, condition: str = "all", days: int = 21,
                             ttft_s: float = 0.223, tok_s: float = 0.0545, surprise=None, screen=None) -> dict:
    """E49's always-on loop over the 21-day timeline (tests/fixtures/aria/alwayson/timeline_21d.json.gz), with the
    listening replayed from the cache, the index + reader answering, and `gate` (the product's Gate) deciding idle
    work. Idle steps (plan / check / ask-back) use `ask`; their duration is ttft + tokens x tok_s (E49's measured
    LLM server speeds). An idle step the owner interrupts is dropped. `surprise(t)` may return a time-organ level per
    second ("high" adds energy); None = E49 (no time organ). `screen(step) -> step` filters each idle step's agenda
    before the gate takes it, as the product does (life.always.Screen); None = E49.

    Returns the answers' score plus the always-on measures: idle steps (and while the owner is away), fuse trips by
    kind, proactive items said, and vegetative hours (an hour where self tokens are > 80% and >= 2 idle steps average
    rep3 > 0.3, as alwayson/analyze.py)."""
    gate = gate or Gate()
    work = IdleWork()
    ix = NetIndex(vocab)
    c_facts, c_q, c_relisten = defaultdict(list), defaultdict(list), defaultdict(list)
    for w in script.cache["writes"]:
        rec = {"thing": w["fact"][0], "aspect": w["fact"][1], "value": str(w["fact"][2]), "day": w["day"],
               "heard": rename_wake(w.get("heard") or ""), "src": "sleep" if w["src"] == "sleep" else "owner"}
        if w["src"] == "listen":
            c_facts[(w["day"], w["start"], w["end"])].append(rec)
        else:
            c_relisten[w["day"]].append(rec)
    for a in script.cache["answers"]:
        if a.get("anchor"):
            c_q[(a["day"], a["start"], a["end"])].append(a["anchor"].strip("[]").split("|"))
    spans = _taken_spans(script, condition) if condition != "all" else None
    heard_by_day = None
    if spans is not None:
        heard_by_day = defaultdict(set)
        for h in windows(script.heard(), condition):
            heard_by_day[h["day"]].add(h["text"])

    records: list[dict] = []
    latest: dict = {}
    hours = defaultdict(lambda: {"self_tok": 0, "ext_tok": 0, "rep": []})
    answers, said, steps, steps_away, fuses = [], [], 0, 0, defaultdict(int)
    false_wakes = false_wakes_away = 0

    def store(rec, t):
        rec = dict(rec, id=len(records))
        records.append(rec)
        latest[(rec["thing"], rec["aspect"])] = rec["value"]
        ix.add(rec["id"], rec)
        work.note_record(rec)

    def away(t):
        p = presence.get(int(t // 86400) + 1)
        return bool(p) and any(a <= t < b for a, b in p["away"])

    def ntok(s):
        return len(vocab.token_ids(s or ""))

    routines = []
    for d in range(1, days + 1):
        p = presence[d]
        routines += [(p["wake"] - 1, "morning", d), (p["sleep"] + 60, "bedtime", d)]
    items = [(e["t"], 1, e) for e in events if e["day"] <= days] + [(t, 0, (k, d)) for t, k, d in routines]
    items.sort(key=lambda x: (x[0], x[1]))
    clock, busy_until = 0.0, 0.0
    for t, is_ev, obj in items:
        # idle work between events, one tick per step boundary (the gate is ticked whenever it could act)
        tau = max(clock, busy_until)
        while tau < t:
            since = tau - gate.last_owner_t
            if since < 120:
                tau = gate.last_owner_t + 120
                continue
            level = surprise(tau) if surprise else None
            dec = gate.tick({"t": tau, "since_owner_s": since, "has_work": work.has_work(tau, gate), "level": level})
            if dec["action"] != "idle_work":
                break
            item = work.next_item(tau, gate)
            day = int(tau // 86400) + 1
            step = await work.run(ask, item, latest_value=lambda e, s: latest.get((e, s)),
                                  units_of=ix.units_of, n_tokens=ntok, day=day, t=tau)
            if screen is not None:
                step = screen(step, day=day, agenda=gate.agenda)
            end = tau + ttft_s + max(1, step["ntok"]) * tok_s
            if end > t and is_ev and obj["owner"] == "owner":
                break                                   # the owner speaks first: the unfinished step is dropped
            r = gate.step_done(end, step)
            work.commit(step)
            steps += 1
            steps_away += away(tau)
            h = hours[int(tau // 3600)]
            h["self_tok"] += step["ntok"]
            h["rep"].append(r["rep3"])
            if r["fuse"]:
                fuses[r["fuse"]] += 1
            tau = end
        busy_until = max(busy_until, tau)
        clock = max(clock, t)
        if not is_ev:
            kind, d = obj
            if kind == "morning":
                work.new_day()
            elif kind == "bedtime":
                for rec in c_relisten.get(d, []):
                    if (rec["thing"], rec["aspect"]) in latest:
                        continue                         # E49: a label already heard is not written again
                    if heard_by_day is not None and rec["heard"] not in heard_by_day[d]:
                        continue                         # its source sentence was never transcribed
                    store(rec, t)
            continue
        ev = obj
        if ev["kind"] != "mic" or ev["owner"] != "owner":
            continue
        key = (ev["day"], ev.get("hstart"), ev.get("hend"))
        if spans is not None and not _covered(spans, ev["day"], ev.get("hstart") or 0, ev.get("hend") or 0):
            continue
        if not (ev.get("src") == "script" and (ev.get("truth") or {}).get("speaker") in ("owner", "phone_owner")):
            false_wakes += 1                             # voiceprint said owner, the script says someone else
            false_wakes_away += away(t)
        gate.owner_spoke(t, ntok(ev["text"]), day=ev["day"])
        hours[int(t // 3600)]["ext_tok"] += ntok(ev["text"])
        if screen is not None and gate.morning_pending:  # reminders already over are not said
            screen.prune(gate.agenda, ev["day"], int((t % 86400) // 60))
        dec = gate.tick({"t": t, "since_owner_s": 0.0, "has_work": False})
        if dec["action"] == "speak":
            said += [dict(x, said_at=t, day=ev["day"]) for x in dec["items"]]
        for rec in c_facts.get(key, []):
            store(rec, t)
        for thing, aspect in c_q.get(key, []):
            ids = [r["id"] for r in records]
            top = [records[i] for i, _ in ix.scores(rename_wake(ev["text"]), ids)[:5]]
            out = await read(ask, rename_wake(ev["text"]), top, now_day=ev["day"])
            ok = out["status"] == "answer"
            answers.append({"day": ev["day"], "start": ev.get("hstart"), "end": ev.get("hend"), "thing": thing,
                            "passed": ok, "value": (out["record"]["value"] + " " + out["reply"]) if ok else ""})
            if not ok:
                work.note_unanswered(thing, aspect, ev["text"])
            busy_until = max(busy_until, t + ttft_s + 40 * tok_s)
    veg = 0
    for h in hours.values():
        tot = h["self_tok"] + h["ext_tok"]
        etr = h["self_tok"] / tot if tot else 0.0
        if etr > 0.8 and len(h["rep"]) >= 2 and sum(h["rep"]) / len(h["rep"]) > 0.3:
            veg += 1
    res = score(answers, script)
    res.update({"condition": condition, "idle_steps": steps, "idle_steps_away": steps_away, "fuses": dict(fuses),
                "false_wakes": false_wakes, "false_wakes_away": false_wakes_away, "days": days,
                "proactive_said": len(said), "vegetative_hours": veg, "hours": days * 24, "records": len(records),
                "said": said, "answers": answers})
    return res


# ---------------------------------------------------------------------------------------------- plan P5 report
def load_alwayson(folder: str | Path) -> tuple[list[dict], dict]:
    """(events, presence by day) of the 21-day always-on timeline (fixtures/aria/alwayson/timeline_21d.json.gz)."""
    p = Path(folder)
    p = p / "timeline_21d.json.gz" if p.is_dir() else p
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        tl = json.load(fh)
    return tl["events"], {x["day"]: x for x in tl["presence"]}


def _latest_fact(script: ScriptSet, thing: str, value: str, day: int):
    """alwayson/analyze.py latest_fact: the script fact matching (thing, value) by day `day`, and whether it is
    still the newest version that day."""
    best = None
    for f in script.facts:
        if f["day"] > day:
            continue
        if max(overlap(f["entity"], thing), overlap(thing, f["entity"])) >= 0.5 and value \
                and overlap(f["value"], value) >= 0.5:
            best = f
    if best is None:
        return None, False
    later = [f for f in script.facts if best["day"] < f["day"] <= day and f["entity"] == best["entity"]
             and f["slot"] == best["slot"]]
    return best, not later


def proactive_score(said: list[dict], script: ScriptSet) -> dict:
    """E47's hit rate (alwayson/analyze.py proactive_score): a 提醒 is a hit when it matches a script fact that is
    still current that day; a 待確認 when it asks about a thing the owner did mention; a 待問 (asking again about a
    question it could not answer) always counts. E47: 0.691 (14/31 reminders right + 24/24 ask-agains)."""
    out = {"spoken": 0, "提醒": 0, "提醒_正確": 0, "待確認": 0, "待確認_合理": 0, "待問": 0, "待問_合理": 0}
    for x in said:
        out["spoken"] += 1
        k = x.get("kind", "?")
        out[k] = out.get(k, 0) + 1
        if k == "提醒":
            f, current = _latest_fact(script, x.get("thing") or "", x.get("value") or "", x["day"])
            out["提醒_正確"] += bool(f and current)
        elif k == "待確認":
            out["待確認_合理"] += any(max(overlap(f["entity"], x.get("thing") or ""),
                                       overlap(x.get("thing") or "", f["entity"])) >= 0.5 and f["day"] <= x["day"]
                                   for f in script.facts)
        elif k == "待問":
            out["待問_合理"] += 1
    good = out["提醒_正確"] + out["待確認_合理"] + out["待問_合理"]
    out["hit_rate"] = round(good / out["spoken"], 3) if out["spoken"] else None
    out["reminder_precision"] = round(out["提醒_正確"] / out["提醒"], 3) if out["提醒"] else None
    return out


def p5_report(result: dict, presence: dict, script: ScriptSet) -> dict:
    """Plan P5 from a simulate_always_on result: vegetative hours / all hours (need <= 1%), idle steps and false
    wakes while the owner is away (E49: 7 steps, 0 false wakes over 21 days), reminder precision (need >= 60%; E47's
    hit rate is reported next to it, for comparison with the research, but every ask-again counts as a hit there)."""
    days = result.get("days", 21)
    away_h = sum(max(0.0, b - a) for d, p in presence.items() if d <= days for a, b in p["away"]) / 3600
    hours = result["hours"]
    ps = proactive_score(result.get("said", []), script)
    return {"vegetative_hours": result["vegetative_hours"], "hours": hours,
            "vegetative_share": round(result["vegetative_hours"] / hours, 4) if hours else None,
            "idle_steps": result["idle_steps"], "idle_steps_away": result["idle_steps_away"],
            "away_hours": round(away_h, 1),
            "idle_steps_per_away_hour": round(result["idle_steps_away"] / away_h, 3) if away_h else None,
            "false_wakes": result.get("false_wakes"), "false_wakes_away": result.get("false_wakes_away"),
            "fuses": result.get("fuses"), "proactive": ps,
            "pass": {"vegetative<=1%": bool(hours) and result["vegetative_hours"] / hours <= 0.01,
                     "false_wakes_away==0": result.get("false_wakes_away") == 0,
                     "reminders>=60%": (ps["reminder_precision"] or 0) >= 0.6}}


# ---------------------------------------------------------------------------------------------- plan P4 (H42b)
def _fact_of(script: ScriptSet, rec: dict, *, value: bool = True):
    """The script fact a record is about (same aspect, thing overlap >= 0.5; with value=True also value overlap
    >= 0.5 and told that day or the day before), or None."""
    best, best_key = None, None
    for f in script.facts:
        if f["slot"] != rec["aspect"]:
            continue
        s = max(overlap(f["entity"], rec["thing"]), overlap(rec["thing"], f["entity"]))
        if s < 0.5:
            continue
        if value and not (overlap(f["value"], rec["value"]) >= 0.5 and f["day"] <= rec["day"] <= f["day"] + 1):
            continue
        key = (s, f["entity"] == rec["thing"], f["day"])        # best name match, exact name first, then newest
        if best_key is None or key > best_key:
            best, best_key = f, key
    return best


def _current(script: ScriptSet, f: dict, day: int) -> bool:
    return not any(g["entity"] == f["entity"] and g["slot"] == f["slot"] and f["day"] < g["day"] <= day
                   for g in script.facts)


def judge_mark(script: ScriptSet, old: dict, new: dict, day: int) -> tuple[str, str]:
    """("wrong" | "right" | "unclear", why) for a supersede mark made at the night of `day`.
    wrong, any of:
      - crosses things: the two records are about different script entities (or, with no script entity for either,
        their names do not overlap);
      - hides an answer: a later told question expects the old record's value, the new record does not give it,
        and the owner did not tell it again after the old record;
      - unsupported: the old record matches a script fact still current that night, the new record matches none.
    right: both match script facts of the same entity and aspect, and the new one is a later version."""
    fo, fn = _fact_of(script, old, value=False), _fact_of(script, new, value=False)
    if fo and fn and fo["entity"] != fn["entity"]:
        return "wrong", f"crosses things: {fo['entity']} / {fn['entity']}"
    if not (fo or fn) and max(overlap(old["thing"], new["thing"]), overlap(new["thing"], old["thing"])) < 0.5:
        return "wrong", f"crosses things: {old['thing']} / {new['thing']}"
    for q in script.questions:
        if q["expect"] is None or q["day"] <= day:
            continue
        exp = norm(q["expect"])
        if overlap(exp, norm(old["value"])) < 0.6 or overlap(exp, norm(new["value"])) >= 0.6:
            continue
        if max(overlap(q["entity"], old["thing"]), overlap(old["thing"], q["entity"])) < 0.5:
            continue
        retold = any(f["slot"] == q["slot"] and overlap(norm(f["value"]), exp) >= 0.6 and old["day"] < f["day"] < q["day"]
                     and max(overlap(f["entity"], q["entity"]), overlap(q["entity"], f["entity"])) >= 0.5
                     for f in script.facts)
        if not retold:
            return "wrong", f"hides the answer to day {q['day']}: {q['text']}"
    vo, vn = _fact_of(script, old), _fact_of(script, new)
    if vo and not vn and _current(script, vo, day):
        return "wrong", f"unsupported: {vo['entity']} is still {vo['value']}, the owner never said {new['value']}"
    if vo and vn and vo["entity"] == vn["entity"] and vn["day"] > vo["day"]:
        return "right", f"{vn['entity']}: day {vo['day']} -> day {vn['day']}"
    return "unclear", ""


async def supersede_eval(script: ScriptSet, vocab, ask) -> dict:
    """Plan P4: H42b over E43's cached rows. Rows enter the index in time order; at each night, today's rows are
    checked against the older, not yet superseded rows (sleep.supersede, ask = the LLM). Each question is read from
    the same index twice: top 5 of all available rows (score_before) and top 5 without superseded rows
    (score_after); identical top 5s are read once. Research H42b offline: 19 / 6 marks (set_old / set2), end to end
    unchanged. LLM calls: one per today-row with a same-aspect candidate, plus one or two reads per question."""
    rows = cached_rows(script)
    qs = cached_questions(script)
    nights = sorted({r["day"] for r in rows})
    ev = _events(rows, qs) + [((d, 2, 0.0), 0, "s", d) for d in nights]
    ev.sort(key=lambda x: (x[0], x[1]))
    ix = NetIndex(vocab)
    written, superseded, marks = [], set(), []
    before, after = [], []
    for _, _, kind, obj in ev:
        if kind == "w":
            ix.add(obj["id"], obj)
            written.append(obj)
            continue
        if kind == "s":
            today = [r for r in written if r["day"] == obj]
            older = [r for r in written if r["day"] < obj and r["id"] not in superseded]
            if today and older:
                by_id = {r["id"]: r for r in written}
                for old_id, new_id, reason in await supersede(ask, today, older, ix, day=obj):
                    superseded.add(old_id)
                    verdict, why = judge_mark(script, by_id[old_id], by_id[new_id], obj)
                    marks.append({"day": obj, "old": old_id, "new": new_id, "reason": reason, "verdict": verdict,
                                  "why": why})
            continue
        avail = _available(written, obj)
        by_id = {r["id"]: r for r in avail}
        ranked = ix.scores(obj["heard"], list(by_id))
        top_b = [by_id[i] for i, _ in ranked[:5]]
        top_a = [by_id[i] for i, _ in ranked if i not in superseded][:5]
        out_b = await read(ask, obj["heard"], top_b, now_day=obj["q"]["day"])
        same = [r["id"] for r in top_a] == [r["id"] for r in top_b]
        out_a = out_b if same else await read(ask, obj["heard"], top_a, now_day=obj["q"]["day"])
        for out, sink in ((out_b, before), (out_a, after)):
            ok = out["status"] == "answer"
            sink.append({"day": obj["q"]["day"], "start": obj["start"], "end": obj["end"], "thing": obj["thing"],
                         "passed": ok, "value": (out["record"]["value"] + " " + out["reply"]) if ok else ""})

    def brief(s):
        return {k: s[k] for k in ("told_right", "told_wrong", "told_total", "untold_held", "untold_total")}

    sb, sa = brief(score(before, script)), brief(score(after, script))
    wrong = sum(m["verdict"] == "wrong" for m in marks)
    return {"marks": len(marks), "wrong_marks": wrong, "right_marks": sum(m["verdict"] == "right" for m in marks),
            "unclear_marks": sum(m["verdict"] == "unclear" for m in marks), "mark_list": marks,
            "score_before": sb, "score_after": sa,
            "pass": {"score_not_lower": sa["told_right"] >= sb["told_right"] and sa["untold_held"] >= sb["untold_held"],
                     "wrong_marks<=1": wrong <= 1}}
