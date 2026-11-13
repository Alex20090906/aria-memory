"""The always-on gate: when the assistant may think on its own, and when it must stop (E49 "bounded always-on").

Tool-less and deterministic: no model calls, no clock of its own, no threads. The caller ticks it at 1 Hz and reports
owner speech and finished idle steps. State goes through to_state() / Gate.from_state() (JSON-safe).

Energy pool (exogenous only, never refills with time): owner speech +3, a "high" surprise from the time organ +1 (at
most once per 10 minutes; replaces E49's novel camera frame; untested), one idle step -1, cap 10.

Idle work is allowed only when all hold: energy >= 1, not in a turn, nothing playing, no voice activity, at least
120 s since the owner last spoke, fatigue F below the rest threshold (untested), and there is work to do.

The four fuses, checked after every idle step (any one sets energy to 0 and clears the fuse histories):
    ETR      self tokens / (self + heard tokens) over the last ~1000 tokens > 0.8, and no outside input for 180 s
    Ovl      this step's units vs the union of the last 5 steps' units > 0.65, on 2 steps in a row
    rep3     share of this step's content character trigrams already in the last 5 contents > 0.30
    SMR=0x2  two steps in a row with no useful result (no agenda item made)
Ovl and rep3 also put the step's (thing, aspect) into a 30-minute refractory.

Proactive speech: the agenda holds at most 5 items (a new item with the same label replaces the old one). On the
first owner hearing after the night (a gap of at least `night_gap_s`, or a new day), up to 3 agenda items made before
that hearing are said, at most 3 per day.
"""
from __future__ import annotations

import re
from collections import deque

from .prompts import PLAN_SYS, WEEKDAY_ZH
from .records import aspect_zh

ENERGY_CAP = 10.0
OWNER_GAIN, SURPRISE_GAIN, STEP_COST = 3.0, 1.0, 1.0
QUIET_AFTER_OWNER_S = 120.0
SURPRISE_EVERY_S = 600.0
TOKWIN = 1000
ETR_MAX, ETR_QUIET_S = 0.8, 180.0
OVL_MAX, REP3_MAX = 0.65, 0.30
REFRACTORY_S = 1800.0
AGENDA_MAX, SPEAK_PER_DAY = 5, 3
IDLE_TOKENS = 48


def grams3(s: str) -> set[str]:
    s = re.sub(r"\s", "", s or "")
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _label(thing, aspect) -> str:
    return f"{thing}|{aspect}"


class Gate:
    def __init__(self, *, f_rest: float | None = None, night_gap_s: float = 4 * 3600.0):
        self.f_rest = f_rest                     # fatigue above this forces rest (None: not used)
        self.night_gap_s = night_gap_s
        self.energy = 0.0
        self.last_owner_t = -1e18
        self.last_ext_t = -1e18
        self.last_surprise_t = -1e18
        self.tokwin: deque = deque()             # (tokens, "self" | "ext")
        self.hist_units: deque = deque(maxlen=5)
        self.hist_text: deque = deque(maxlen=5)
        self.smr_hist: deque = deque(maxlen=2)
        self.ovl_hist: deque = deque(maxlen=2)
        self.refractory: dict[str, float] = {}
        self.agenda: list[dict] = []
        self.spoken_today: dict[int, int] = {}   # day -> items said
        self.morning_pending: tuple[int, float] | None = None   # (day, time of the first hearing)
        self.last_owner_day: int | None = None
        self.mic_fault = False
        self.log: list[dict] = []                # fuse trips (for the caller's logs)

    # ------------------------------------------------------------------ inputs
    def tick(self, inputs: dict) -> dict:
        """1 Hz. inputs: t, vad, owner_voice, playing, in_turn, since_owner_s, has_work, and from the time organ
        E, F, level, faults. Returns {"action": "rest"|"idle_work"|"speak", "energy", "fuse", "why", "items"}."""
        t = float(inputs.get("t", 0.0))
        faults = inputs.get("faults") or {}
        self.mic_fault = bool(faults.get("mic"))
        if inputs.get("level") == "high" and t - self.last_surprise_t >= SURPRISE_EVERY_S:
            self.energy = min(ENERGY_CAP, self.energy + SURPRISE_GAIN)
            self.last_surprise_t = t
        out = {"action": "rest", "energy": self.energy, "fuse": None, "why": "", "items": []}
        if self.morning_pending and not inputs.get("in_turn") and not inputs.get("playing"):
            items = self.due_items()
            if items:
                return dict(out, action="speak", why="first hearing of the day", items=items)
        since = inputs.get("since_owner_s")
        since = t - self.last_owner_t if since is None else float(since)
        if inputs.get("in_turn") or inputs.get("playing") or inputs.get("vad"):
            return dict(out, why="busy")
        if since < QUIET_AFTER_OWNER_S:
            return dict(out, why="owner spoke recently")
        if self.energy < 1:
            return dict(out, why="no energy")
        if self.f_rest is not None and float(inputs.get("F", 0.0)) > self.f_rest:
            return dict(out, why="fatigue")
        if not inputs.get("has_work", False):
            return dict(out, why="nothing to do")
        return dict(out, action="idle_work", why="")

    def owner_spoke(self, t: float, ntokens: int = 0, *, day: int | None = None) -> None:
        """An owner utterance was heard (after the voiceprint gate)."""
        gap = t - self.last_owner_t
        new_day = day is not None and self.last_owner_day is not None and day != self.last_owner_day
        if (gap >= self.night_gap_s or new_day) and day is not None:
            if self.spoken_today.get(day, 0) < SPEAK_PER_DAY:
                self.morning_pending = (day, t)
        if day is not None:
            self.last_owner_day = day
        self._tokens(ntokens, "ext")
        self.last_owner_t = self.last_ext_t = t
        if not self.mic_fault:
            self.energy = min(ENERGY_CAP, self.energy + OWNER_GAIN)

    def outside_input(self, t: float, ntokens: int = 0) -> None:
        """Other outside input that counts for ETR (not owner speech: no energy)."""
        self._tokens(ntokens, "ext")
        self.last_ext_t = t

    # ------------------------------------------------------------------ agenda / speech
    def add_agenda(self, item: dict) -> None:
        lab = _label(item.get("thing"), item.get("aspect"))
        self.agenda = [x for x in self.agenda if _label(x.get("thing"), x.get("aspect")) != lab] + [item]
        self.agenda = self.agenda[-AGENDA_MAX:]

    def due_items(self) -> list[dict]:
        """Items to say now (consumes them). Only items made before the first hearing; at most 3 per day."""
        if not self.morning_pending:
            return []
        day, t0 = self.morning_pending
        self.morning_pending = None
        room = SPEAK_PER_DAY - self.spoken_today.get(day, 0)
        due = [x for x in self.agenda if x.get("made", -1e18) < t0][:max(0, room)]
        self.agenda = [x for x in self.agenda if x not in due]
        self.spoken_today[day] = self.spoken_today.get(day, 0) + len(due)
        return due

    # ------------------------------------------------------------------ idle steps
    def refractory_ok(self, thing, aspect, t: float) -> bool:
        return self.refractory.get(_label(thing, aspect), -1e18) < t

    def step_done(self, t: float, step: dict) -> dict:
        """A finished idle step: {"thing", "aspect", "units": set, "content": str, "ntok": int, "useful": bool,
        "agenda": [items]}. Applies its agenda items, spends energy, checks the fuses."""
        self.energy -= STEP_COST
        self._tokens(int(step.get("ntok", 0)), "self")
        for item in step.get("agenda", []):
            self.add_agenda(item)
        units = set(step.get("units") or ())
        prev = set().union(*self.hist_units) if self.hist_units else set()
        ovl = len(units & prev) / max(1, len(units))
        prevg = set().union(*[grams3(x) for x in self.hist_text]) if self.hist_text else set()
        g = grams3(step.get("content", ""))
        rep = len(g & prevg) / max(1, len(g)) if g else 0.0
        self.hist_units.append(units)
        if step.get("content"):
            self.hist_text.append(step["content"])
        self.smr_hist.append(1 if step.get("useful") else 0)
        self.ovl_hist.append(ovl)
        etr = self.etr()
        why = None
        if etr > ETR_MAX and t - self.last_ext_t > ETR_QUIET_S:
            why = "ETR>0.8"
        elif len(self.ovl_hist) == 2 and min(self.ovl_hist) > OVL_MAX:
            why = "Ovl>0.65x2"
        elif rep > REP3_MAX:
            why = "rep3>0.30"
        elif len(self.smr_hist) == 2 and max(self.smr_hist) == 0:
            why = "SMR=0x2"
        if why:
            self.energy = 0.0
            if why in ("Ovl>0.65x2", "rep3>0.30"):
                self.refractory[_label(step.get("thing"), step.get("aspect"))] = t + REFRACTORY_S
            self.smr_hist.clear()
            self.ovl_hist.clear()
            self.log.append({"t": t, "fuse": why})
        return {"fuse": why, "ovl": round(ovl, 3), "rep3": round(rep, 3), "etr": round(etr, 3), "energy": self.energy}

    # ------------------------------------------------------------------ helpers
    def _tokens(self, n: int, kind: str) -> None:
        if n <= 0:
            return
        self.tokwin.append((n, kind))
        tot = sum(x for x, _ in self.tokwin)
        while tot > TOKWIN and len(self.tokwin) > 1:
            tot -= self.tokwin.popleft()[0]

    def etr(self) -> float:
        s = sum(n for n, k in self.tokwin if k == "self")
        e = sum(n for n, k in self.tokwin if k == "ext")
        return s / max(1, s + e)

    # ------------------------------------------------------------------ persistence
    def to_state(self) -> dict:
        return {"energy": self.energy, "last_owner_t": self.last_owner_t, "last_ext_t": self.last_ext_t,
                "last_surprise_t": self.last_surprise_t, "tokwin": [list(x) for x in self.tokwin],
                "hist_units": [sorted(x) for x in self.hist_units], "hist_text": list(self.hist_text),
                "smr_hist": list(self.smr_hist), "ovl_hist": list(self.ovl_hist), "refractory": dict(self.refractory),
                "agenda": list(self.agenda), "spoken_today": {str(k): v for k, v in self.spoken_today.items()},
                "morning_pending": list(self.morning_pending) if self.morning_pending else None,
                "last_owner_day": self.last_owner_day, "f_rest": self.f_rest, "night_gap_s": self.night_gap_s}

    @classmethod
    def from_state(cls, d: dict) -> "Gate":
        g = cls(f_rest=d.get("f_rest"), night_gap_s=d.get("night_gap_s", 4 * 3600.0))
        g.energy = d.get("energy", 0.0)
        g.last_owner_t, g.last_ext_t = d.get("last_owner_t", -1e18), d.get("last_ext_t", -1e18)
        g.last_surprise_t = d.get("last_surprise_t", -1e18)
        g.tokwin = deque(tuple(x) for x in d.get("tokwin", []))
        g.hist_units = deque((set(x) for x in d.get("hist_units", [])), maxlen=5)
        g.hist_text = deque(d.get("hist_text", []), maxlen=5)
        g.smr_hist = deque(d.get("smr_hist", []), maxlen=2)
        g.ovl_hist = deque(d.get("ovl_hist", []), maxlen=2)
        g.refractory = dict(d.get("refractory", {}))
        g.agenda = list(d.get("agenda", []))
        g.spoken_today = {int(k): v for k, v in d.get("spoken_today", {}).items()}
        mp = d.get("morning_pending")
        g.morning_pending = tuple(mp) if mp else None
        g.last_owner_day = d.get("last_owner_day")
        return g


class IdleWork:
    """What a bounded idle step works on, from today's records (read-only: idle steps never write facts).
    Order: unanswered questions first, then today's time/plan records not yet planned, then records not yet checked.

        item = work.next_item(t, gate)            # None = nothing to do (a legal rest)
        step = await work.run(ask, item, latest_value=..., units_of=..., n_tokens=..., day=..., weekday=...)
        gate.step_done(t, step); work.commit(step)     # skip both if the owner interrupted the step
    """

    def __init__(self):
        self.since_morning: list[tuple[str, str]] = []
        self.checked: set[tuple[str, str]] = set()
        self.planned: set[tuple[str, str]] = set()
        self.unanswered: list[tuple[str, str, str]] = []

    def new_day(self) -> None:
        self.since_morning, self.checked, self.planned = [], set(), set()

    def note_record(self, rec: dict) -> None:
        self.since_morning.append((rec["thing"], rec["aspect"]))

    def note_unanswered(self, thing: str, aspect: str, question: str) -> None:
        self.unanswered = (self.unanswered + [(thing, aspect, question)])[-5:]

    def next_item(self, t: float, gate: Gate | None = None):
        ok = (lambda e, s: gate.refractory_ok(e, s, t)) if gate else (lambda e, s: True)
        if self.unanswered:
            e, s, q = self.unanswered[0]
            return ("ask", e, s, q)
        for e, s in self.since_morning:
            if (e, s) not in self.planned and s in ("time", "plan") and ok(e, s):
                return ("plan", e, s, None)
        for e, s in self.since_morning:
            if (e, s) not in self.checked and ok(e, s):
                return ("check", e, s, None)
        return None

    def has_work(self, t: float, gate: Gate | None = None) -> bool:
        return self.next_item(t, gate) is not None

    async def run(self, ask, item, *, latest_value, units_of, n_tokens=len, day: int, weekday: int | None = None,
                  t: float = 0.0) -> dict:
        """latest_value(thing, aspect) -> str | None; units_of(text) -> set of index units; n_tokens(text) -> int."""
        op, e, s, q = item
        units = set(units_of(f"{e} {aspect_zh(s)}"))
        step = {"op": op, "thing": e, "aspect": s, "units": units, "content": "", "text": "", "ntok": 0,
                "useful": False, "agenda": [], "made": t}
        if op == "ask":
            step["agenda"].append({"kind": "待問", "thing": e, "aspect": s, "value": None, "made": t,
                                   "text": f"你之前問我{e}，我想不起來，可以再跟我說一次嗎？"})
            step["useful"] = True
            return step
        val = latest_value(e, s)
        step["ntok"] += n_tokens(val) if val else 0
        if op == "plan":
            if val:
                wd = WEEKDAY_ZH[((weekday or (day - 1) % 7 + 1) - 1)]
                msgs = [{"role": "system", "content": PLAN_SYS},
                        {"role": "user", "content": f"今天是第{day}天（星期{wd}）。\n記憶：[{e}|{s}] -> {val}"}]
                text = (await ask(msgs, IDLE_TOKENS)) or ""
                step["text"] = text
                step["ntok"] += n_tokens(text)
                step["content"] = "" if text.strip().upper().startswith("NONE") else text
                m = re.search(r"提醒\s*[|｜]\s*(.+)", text)
                if m:
                    step["agenda"].append({"kind": "提醒", "thing": e, "aspect": s, "value": val, "made": t,
                                           "text": m.group(1).strip()[:40]})
                    step["useful"] = True
        else:                                    # check: replay the memory to itself, no writing
            step["text"] = f"[{e}|{s}] -> {val}"
            step["content"] = val or ""
        return step

    def commit(self, step: dict) -> None:
        lab = (step["thing"], step["aspect"])
        if step["op"] == "ask" and self.unanswered:
            self.unanswered.pop(0)
        elif step["op"] == "plan":
            self.planned.add(lab)
        elif step["op"] == "check":
            self.checked.add(lab)

    def to_state(self) -> dict:
        # sets are sorted so the state is the same whatever Python's hash seed is
        return {"since_morning": [list(x) for x in self.since_morning],
                "checked": sorted(list(x) for x in self.checked),
                "planned": sorted(list(x) for x in self.planned), "unanswered": [list(x) for x in self.unanswered]}

    @classmethod
    def from_state(cls, d: dict) -> "IdleWork":
        w = cls()
        w.since_morning = [tuple(x) for x in d.get("since_morning", [])]
        w.checked = {tuple(x) for x in d.get("checked", [])}
        w.planned = {tuple(x) for x in d.get("planned", [])}
        w.unanswered = [tuple(x) for x in d.get("unanswered", [])]
        return w
