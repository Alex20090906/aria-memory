"""H32 offline: does the listener route owner questions to the answerer, and does it route statements by mistake?

E42 "not routed" questions heard as the owner were lost two ways (route_diag.py):
1. the listener emitted a question with a slot outside the list ("Q 教練 | name", "Q 貓鯊 | brand"), which zh_v3
   parse() drops; retrieve-then-read does not need the slot, so any question line can be routed;
2. speech recognition turned 哪 into 拿 ("放哪去了" -> "放拿去了") or a final 咧 into 列/臉, and the listener wrote a
   statement (often a junk fact, "location | 拿去").
Variant h32 = (1) accept any question line + (2) one listening rule about these common mishearings, phrased in general
terms (no script sentence is quoted).

Replays the listening call for every owner-heard sentence of both script sets against the normal LLM server service, with
TOPIC carried forward from the replayed outputs and an empty staging list (the run's staging state is not known
offline), so both variants see the same approximation.

Counts: told / never-told question segments that get a question line; owner sentences that are not questions but get
a question line (false routing); junk location facts whose value is a mishearing of 哪.

    python listen_route_eval.py <zh_scripts.json> <timeline.json> <asr.json> [...]
env: LR_VARIANTS (default "base,h32"), LLM_API_KEY
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from route_diag import NS, chat_shots, WD  # noqa: E402

SLOTS = NS["SLOTS"]
RULE = ("7. 語音辨識常把「哪」聽成「拿」，把句尾的「咧」聽成「列」「臉」「了」。如果把這些字換回來，這句話是在問 Aria"
        "某個東西在哪、怎麼了、是什麼，就輸出 Q 行，不要把「拿去」「拿裡」當成內容記下來。\n")
Q_ANY = re.compile(r"^Q\s*(.+)$")


def questions(lines, any_slot):
    qs = []
    for line in lines:
        line = line.strip()
        if not line.startswith("Q"):
            continue
        if line.count("|") == 1:
            e, s = [x.strip() for x in line[1:].split("|")]
            if e and (s in SLOTS or any_slot):
                qs.append((e, s))
        elif any_slot and Q_ANY.match(line) and "|" not in line:
            qs.append((Q_ANY.match(line).group(1).strip(), "?"))
    return qs


def junk_na(lines):
    n = 0
    for line in lines:
        if line.count("|") == 2:
            e, s, v = [x.strip() for x in line.split("|")]
            if s == "location" and re.fullmatch(r"(拿|哪)(去|裡|邊)?(了|啦|哦|喔)?", v):
                n += 1
    return n


def run(sc_path, tl_path, asr_path, variant):
    sc = json.load(open(sc_path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(tl_path, encoding="utf-8"))}
    asr = json.load(open(asr_path, encoding="utf-8"))
    sysmsg = NS["LISTEN_SYS"]
    if variant == "h32":
        sysmsg = sysmsg.replace("輸出格式（每行一項）：", RULE + "輸出格式（每行一項）：")
        assert RULE in sysmsg
    qsegs = {}
    for q in sc["questions"]:
        seg = next((g for g in tl[q["day"]]["segments"] if g["text"] == q["text"]), None)
        if seg:
            qsegs.setdefault(q["day"], []).append((seg, q["expect"] is not None))
    told_r = told_n = never_r = never_n = false_q = stmts = junk = 0
    for day in asr:
        dn = day["day"]
        dd = f"今天是第{dn}天（星期{WD[(dn - 1) % 7]}）。"
        topic, ctx = "（無）", []
        hit = {}
        for h in day["heard"]:
            ctx = (ctx + [("[使用者] " if h["owner"] == "owner" else "[其他人] ") + h["text"]])[-6:]
            if h["owner"] != "owner":
                continue
            text = (dd + f"\nTOPIC：{topic}\n前文：\n" + ("\n".join(ctx[:-1]) or "（無）") + "\n暫存：\n（無）"
                    + "\n剛聽到的一句：\n" + ctx[-1])
            out = chat_shots(sysmsg + NS["STAGE_SYS"], NS["LISTEN_SHOTS"], text)
            for line in out:
                if line.upper().startswith("TOPIC"):
                    topic = line[5:].strip(" ：:") or topic
            qs = questions(out, variant == "h32")
            junk += junk_na(out)
            mine = [i for i, (g, _) in enumerate(qsegs.get(dn, []))
                    if min(h["end"], g["end"]) - max(h["start"], g["start"]) > 0.3]
            if mine:
                for i in mine:
                    hit[i] = hit.get(i, False) or bool(qs)
            else:
                stmts += 1
                if qs:
                    false_q += 1
                    print(f"   false Q day {dn}: {h['text'][:40]} -> {qs}", flush=True)
        for i, (g, told) in enumerate(qsegs.get(dn, [])):
            r = hit.get(i, False)
            if told:
                told_n += 1
                told_r += r
            else:
                never_n += 1
                never_r += r
    print(f"  {variant}: told questions routed {told_r}/{told_n} | never-told routed {never_r}/{never_n} | statements "
          f"routed as questions {false_q}/{stmts} | junk 拿/哪 location facts {junk}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    for v in os.environ.get("LR_VARIANTS", "base,h32").split(","):
        for sc, tl, asr in zip(a[0::3], a[1::3], a[2::3]):
            print("==", sc, flush=True)
            run(sc, tl, asr, v)
