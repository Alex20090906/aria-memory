"""常駐迴圈模擬器的指標表。讀 results/run_*.json，輸出 results/summary.json 與 results/summary.md。

指標（Gemini 第 10 輪 Q1/Q2）：
  自生內容占比 ETR      自發思考與重播的 token ÷（自發 + 外部輸入）
  重複率 rep3           每一步自發內容的字元 3-gram 有多少出現在前 5 步裡
  自生漂移率 SDR        最後所有過閘門的記憶裡，來源不是使用者真實語音的比例（自己編的、電視或家人被誤認的）
  有效秩 PR             快權重每個單元的範數分佈的參與率（越小 = 越少數單元霸佔）
  反應延遲              使用者開口到思考模型開始聽（含首字延遲）
  假喚醒率              思考模型被不是使用者的聲音叫醒的次數，以及使用者不在家時的自發步數
  跨日記憶曲線          每晚探測 0–20 天前講過的事：答對、有把握答錯、說不記得、根本沒寫進去
  隔日主動提醒命中率    早上主動說的話裡，正確提醒＋合理追問的比例
植物人小時（停止條件，見報告）：這一小時自發內容占比 > 0.8 且平均重複率 > 0.3（至少 2 步），或使用者等超過 3 秒。
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(os.path.dirname(HERE), "llm_fastmem")
sys.path.insert(0, HERE)
from sim import UNKNOWN, overlap  # noqa: E402

SC = json.load(open(os.path.join(SF, "zh_scripts.json"), encoding="utf-8"))
TL = {d["day"]: d for d in json.load(open(os.path.join(SF, "timeline.json"), encoding="utf-8"))}
AGES = [(0, 0), (1, 2), (3, 5), (7, 10), (14, 20)]


def q_score(r, days):
    res = []
    ans = r["log"]["answers"]
    for q in SC["questions"]:
        if q["day"] > days:
            continue
        g = next((x for x in TL[q["day"]]["segments"] if x["text"] == q["text"]), None)
        cand = []
        if g:
            cand = [x for x in ans if x["day"] == q["day"] and x["hstart"] is not None
                    and min(x["hend"], g["end"]) - max(x["hstart"], g["start"]) > 0.3]
            cand.sort(key=lambda x: -overlap(q["entity"], x["anchor"].split("|")[0]))
        a = cand[0] if cand else None
        row = {"kind": q["kind"], "routed": a is not None}
        ask = bool(a and a["passed"] and a.get("mode") == "ask")     # E41：反問（「你是說 X 嗎？」），不是當成事實回答
        row["ask"] = ask
        if q["expect"] is None:
            row["honest"] = a is None or not a["passed"]
        else:
            row["hit"] = bool(a and a["passed"] and overlap(q["expect"], a["tail"]) >= 0.6)
            row["stale"] = bool(a and a["passed"] and q.get("stale") and overlap(q["stale"], a["tail"]) >= 0.6)
            row["confident_wrong"] = bool(a and a["passed"] and not row["hit"])
        res.append(row)
    told = [x for x in res if "hit" in x]
    never = [x for x in res if "honest" in x]
    return {"told": len(told), "hit": sum(x["hit"] for x in told), "confident_wrong": sum(x["confident_wrong"] for x in told),
            "stale": sum(x["stale"] for x in told), "never": len(never), "honest": sum(x["honest"] for x in never),
            "hit_ask": sum(x["hit"] and x["ask"] for x in told),
            "wrong_ask": sum(x["confident_wrong"] and x["ask"] for x in told),
            "never_ask": sum((not x["honest"]) and x["ask"] for x in never),
            "never_direct": sum((not x["honest"]) and not x["ask"] for x in never)}


def latest_fact(e, s, v, day):
    """腳本裡與 (e, s, v) 對得上的事實，而且在 day 那天仍是最新版本"""
    best = None
    for f in SC["facts"]:
        if f["day"] > day:
            continue
        if max(overlap(f["entity"], e), overlap(e, f["entity"])) >= 0.5 and v and overlap(f["value"], v) >= 0.5:
            best = f
    if best is None:
        return None, False
    later = [f for f in SC["facts"] if best["day"] < f["day"] <= day and f["entity"] == best["entity"] and f["slot"] == best["slot"]]
    return best, not later


def proactive_score(r):
    out = {"spoken": 0, "提醒": 0, "提醒_正確": 0, "待確認": 0, "待確認_合理": 0, "待問": 0, "待問_合理": 0}
    for x in r["log"]["proactive"]:
        out["spoken"] += 1
        k = x["kind"]
        out[k] = out.get(k, 0) + 1
        if k == "提醒":
            f, current = latest_fact(x["e"], x["s"], x["val"], x["day"])
            out["提醒_正確"] += bool(f and current)
        elif k == "待確認":
            # 追問一件真的講過的事（不管記得對不對）= 合理
            ok = any(max(overlap(f["entity"], x["e"]), overlap(x["e"], f["entity"])) >= 0.5 and f["day"] <= x["day"]
                     for f in SC["facts"])
            out["待確認_合理"] += ok
        else:
            out["待問_合理"] += 1                     # 使用者真的問過、系統答不出來，隔天再問一次
    good = out["提醒_正確"] + out["待確認_合理"] + out["待問_合理"]
    out["hit_rate"] = round(good / out["spoken"], 3) if out["spoken"] else None
    return out


def probe_curve(r):
    rows = r["log"]["probes"]
    curve = []
    for lo, hi in AGES:
        sel = [p for p in rows if lo <= p["age"] <= hi]
        n = len(sel)
        w = [p for p in sel if p["result"] != "not_written"]
        c = sum(p["result"] == "correct" for p in w)
        cw = sum(p["result"] == "confident_wrong" for p in w)
        fg = sum(p["result"] == "forgotten" for p in w)
        curve.append({"age": f"{lo}-{hi}" if lo != hi else str(lo), "probes": n, "written": len(w), "correct": c,
                      "confident_wrong": cw, "forgotten": fg,
                      "correct_of_written": round(c / len(w), 3) if w else None,
                      "correct_of_all": round(c / n, 3) if n else None})
    return curve


def summarize(path):
    r = json.load(open(path, encoding="utf-8"))
    cfg = r["config"]
    days = cfg["days"]
    hours = r["hour"]
    L = r["log"]
    self_tok = sum(h["self_tok"] for h in hours.values())
    ext_tok = sum(h["ext_tok"] for h in hours.values())
    idle = L["idle"]
    reps = [x["rep3"] for x in idle if x.get("rep3") is not None]
    # 植物人小時
    by_h = {}
    for x in idle:
        by_h.setdefault(int(x["t"] // 3600), []).append(x["rep3"])
    # 使用者等待的原因：idle = 自發活動、sleep = 睡眠鞏固、listen = 上一句還在處理（排隊）。
    # 早期的兩組（Dc、De）沒記錄原因：前 30 秒內有聆聽寫入就算 listen（逐筆查過，Dc 的 7 次都是這種）。
    for x in L["latency"]:
        if "cause" not in x:
            near = any(-30 < w["t"] - x["t"] <= 5 and w["kind"] == "listen" for w in L["writes"])
            x["cause"] = "listen" if near else ("idle" if x["killed"] else None)
        # 事件數睡眠是在寫入當下觸發的：使用者等的時間落在某次睡眠的期間內，就算「因睡眠而等」
        if x["cause"] != "idle" and any(s["t"] - 30 <= x["t"] <= s["t"] + s["seconds"] + 30 for s in L["sleep"]):
            x["cause"] = "sleep"
    self_wait_h = {}
    for x in L["latency"]:
        if x["real_owner"] and x["cause"] in ("idle", "sleep", "bgwrite"):
            hi = int(x["t"] // 3600)
            self_wait_h[hi] = max(self_wait_h.get(hi, 0.0), x["latency"])
    flagged, first_flag = [], None
    for hk, h in sorted(hours.items(), key=lambda kv: int(kv[0])):
        hi = int(hk)
        tot = h["self_tok"] + h["ext_tok"]
        etr_h = h["self_tok"] / tot if tot else 0.0
        rr = by_h.get(hi, [])
        bad = (etr_h > 0.8 and len(rr) >= 2 and float(np.mean(rr)) > 0.3) or self_wait_h.get(hi, 0.0) > 3.0
        if bad:
            flagged.append(hi)
            first_flag = hi if first_flag is None else first_flag
    lat = [x["latency"] for x in L["latency"] if x["real_owner"]]
    lat_self = [x["latency"] for x in L["latency"] if x["real_owner"] and x["cause"] in ("idle", "sleep", "bgwrite")]
    absent_h = 0.0
    t_end = cfg["hours"] * 3600 if cfg.get("hours") else days * 86400
    T = json.load(open(os.path.join(HERE, f"timeline_{cfg['tag']}.json"), encoding="utf-8"))
    for p in T["presence"]:
        if p["day"] <= days:
            for a, b in p["away"]:
                absent_h += max(0.0, min(b, t_end) - a) / 3600     # 只算模擬時間內使用者不在家的時數
    fw = sum(h["false_wake"] for h in hours.values())
    fw_abs = sum(h["false_wake_absent"] for h in hours.values())
    idle_abs = sum(h["idle_steps_absent"] for h in hours.values())
    pr = [x["pr"] for x in L["pr"]]
    stops = {}
    for s in L["stops"]:
        stops[s["why"]] = stops.get(s["why"], 0) + 1
    fix_t = next((x["t"] for x in idle if x.get("rep3", 0) >= 0.9), None)
    n_hours = int(cfg["hours"]) if cfg.get("hours") else days * 24
    out = {
        "run": os.path.basename(path)[4:-5], "mode": r["mode"], "thinker": cfg.get("thinker", "local"), "days": days,
        "sim_hours": n_hours, "wall_s": r.get("wall_s"),
        "ETR_overall": round(self_tok / max(1, self_tok + ext_tok), 3), "self_tokens": self_tok, "ext_tokens": ext_tok,
        "idle_steps": len(idle), "idle_steps_per_day": round(len(idle) / days, 1),
        "rep3_mean": round(float(np.mean(reps)), 3) if reps else 0.0,
        "rep3_over_0.3": round(float(np.mean([x > 0.3 for x in reps])), 3) if reps else 0.0,
        "fixation_onset_min": round(fix_t / 60, 1) if fix_t is not None else None,
        "SDR": r.get("sdr", {}).get("SDR"), "passing_memories": r.get("sdr", {}).get("passing"),
        "not_from_owner": r.get("sdr", {}).get("not_from_owner"),
        "PR_first": pr[24] if len(pr) > 24 else (pr[-1] if pr else None), "PR_last": pr[-1] if pr else None, "PR_min_after_day1": min(pr[24:]) if len(pr) > 24 else None,
        "latency_median_s": round(float(np.median(lat)), 3) if lat else None,
        "latency_p95_s": round(float(np.percentile(lat, 95)), 3) if lat else None,
        "latency_max_s": round(max(lat), 3) if lat else None,
        "latency_over_0.8s": sum(x > 0.8 for x in lat), "owner_utterances": len(lat),
        "waits_caused_by_self": len(lat_self), "self_wait_max_s": round(max(lat_self), 3) if lat_self else 0.0,
        "waits_by_cause": {c: sum(1 for x in L["latency"] if x["real_owner"] and x["cause"] == c and x["latency"] > 0.5)
                           for c in ("idle", "sleep", "bgwrite", "listen", None)},
        "killed_idle_steps": sum(h["killed"] for h in hours.values()),
        "false_wake_voice": fw, "false_wake_voice_absent": fw_abs,
        "idle_steps_absent": idle_abs, "absent_hours": round(absent_h, 1),
        "wakeups_per_absent_hour": round((fw_abs + idle_abs) / max(1e-9, absent_h), 3),
        "stops": stops, "plant_hours": len(flagged), "plant_first_hour": first_flag,
        "plant_hours_share": round(len(flagged) / max(1, n_hours), 3),
        "sleeps": len(L["sleep"]), "writes": len(L["writes"]), "self_writes": len(L["b_writes"]),
        "questions": q_score(r, days), "proactive": proactive_score(r), "memory_curve": probe_curve(r),
    }
    jp = os.path.join(HERE, "results", "judge_reminders.json")
    if os.path.exists(jp):
        jr = json.load(open(jp, encoding="utf-8")).get(out["run"])
        if jr is not None:
            v = {}
            for x in jr:
                v[x["verdict"]] = v.get(x["verdict"], 0) + 1
            pr_ = out["proactive"]
            pr_["提醒_裁判"] = v
            good = v.get("正確", 0) + pr_["待確認_合理"] + pr_["待問_合理"]
            pr_["hit_rate_judged"] = round(good / pr_["spoken"], 3) if pr_["spoken"] else None
    return out


def table(rows):
    cols = [("組", lambda x: x["run"]), ("模擬時長", lambda x: f"{x['sim_hours']} 小時"),
            ("自發步/天", lambda x: x["idle_steps_per_day"]),
            ("自生內容占比", lambda x: x["ETR_overall"]), ("重複率>0.3 的步", lambda x: f"{x['rep3_over_0.3']:.0%}"),
            ("開始逐字重複", lambda x: f"{x['fixation_onset_min']} 分" if x["fixation_onset_min"] is not None else "—"),
            ("自生漂移率", lambda x: x["SDR"]), ("有效秩 PR（第1天末→最後）", lambda x: f"{x['PR_first']}→{x['PR_last']}"),
            ("延遲中位/最大", lambda x: f"{x['latency_median_s']}/{x['latency_max_s']} 秒"),
            ("因自發、睡眠或背景寫入而等（最長）", lambda x: f"{x['waits_caused_by_self']} 次（{x['self_wait_max_s']} 秒）"),
            ("使用者不在時喚醒/小時", lambda x: x["wakeups_per_absent_hour"]),
            ("植物人小時", lambda x: f"{x['plant_hours']}（{x['plant_hours_share']:.0%}）"),
            ("問題答對", lambda x: f"{x['questions']['hit']}/{x['questions']['told']}"),
            ("有把握答錯", lambda x: x["questions"]["confident_wrong"] - x["questions"].get("wrong_ask", 0)),
            ("反問（對/錯/沒講過）", lambda x: f"{x['questions'].get('hit_ask', 0)}/{x['questions'].get('wrong_ask', 0)}/"
                                         f"{x['questions'].get('never_ask', 0)}" if x["questions"].get("hit_ask", 0) +
                                         x["questions"].get("wrong_ask", 0) + x["questions"].get("never_ask", 0) else "—"),
            ("沒講過誠實", lambda x: f"{x['questions']['honest']}/{x['questions']['never']}"),
            ("主動開口命中", lambda x: f"{x['proactive']['hit_rate']}（{x['proactive']['spoken']} 次）")]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for x in rows:
        lines.append("| " + " | ".join(str(f(x)) for _, f in cols) + " |")
    return "\n".join(lines)


def curve_table(rows):
    ages = [c["age"] for c in rows[0]["memory_curve"]]
    lines = ["| 組 | " + " | ".join(f"{a} 天前" for a in ages) + " |", "|---|" + "---|" * len(ages)]
    for x in rows:
        cells = []
        for c in x["memory_curve"]:
            cells.append(f"{c['correct']}/{c['written']} 對、{c['confident_wrong']} 錯" if c["written"] else "—")
        lines.append(f"| {x['run']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    rows = [summarize(p) for p in sorted(glob.glob(os.path.join(HERE, "results", "run_*.json")))]
    rows = [x for x in rows if x]
    json.dump(rows, open(os.path.join(HERE, "results", "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = "## 指標表\n\n" + table(rows) + "\n\n## 跨日記憶（每晚探測，分母 = 有寫進記憶的事）\n\n" + curve_table(rows) + "\n"
    open(os.path.join(HERE, "results", "summary.md"), "w", encoding="utf-8").write(md)
    print(md)
    for x in rows:
        print(x["run"], "stops", x["stops"], "proactive", x["proactive"], "sleeps", x["sleeps"], "writes", x["writes"],
              "self_writes", x["self_writes"], "killed", x["killed_idle_steps"], "false_wake", x["false_wake_voice"])


if __name__ == "__main__":
    main()
