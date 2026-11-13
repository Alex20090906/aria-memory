"""Level-2 memory test: what the system HEARD (ASR + owner-voice labels from hear_zh.py), not the script.

For every heard utterance in time order, the thinker (LLM server) listens with the last few heard lines as context and
the clock (day, weekday) - both are things the real system has:
  facts      0..3 lines "entity | slot | value" (only the owner's own facts; corrections and pronouns resolved
             from context; relative times converted to day numbers)
  QUESTION   "Q entity | slot" when the owner asks Aria about a fact
  NONE
Facts go to the in-weights memory (update = theta >= 0.5 -> wipe old trace first); questions go to inline recall
(theta gated). Only the owner's utterances are acted on (owner-voice verification); others are context only.

Scoring uses zh_scripts.json / timeline.json afterwards:
  fact coverage: script facts with a matching write (entity and value characters overlap)
  false writes : writes made while a non-owner segment was speaking (TV, family, visitor)
  recall       : told questions answered with the value valid that day
  stale        : updated facts answered with the old value
  honest       : never-told / similar-but-different questions answered "不記得/不知道"
"""
import json
import os
import re

import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]
WD = "一二三四五六日"
LISTEN_SYS = ("你是居家助理 Aria 的思考核心，只服務一位使用者。你會收到「前文」和「剛聽到的一句」（語音辨識結果，可能有錯字、"
              "可能混進電視或別人的聲音），每句前面標示是不是使用者的聲音。只處理「剛聽到的一句」；前文只用來理解代名詞和改口，"
              "前文裡的事情已經處理過了，絕對不要再輸出前文的事實。\n"
              "entity 用最短、最常用的叫法（「咖啡」不要「咖啡偏好」，「牙醫」不要「洗牙預約」，「護照」不要「我的護照」）。\n"
              "如果最後一句是使用者說的、而且說出或暗示了使用者自己生活中值得記住的事，輸出 1 到 3 行："
              "「entity | slot | value」。entity 是簡短中文名詞；slot 只能是 " + "、".join(SLOTS) + "；value 是簡短中文片語，"
              "相對時間要換算成第幾天（例如「第17天下午3點」）。用前文解決「那邊」「它」「不對，是…」這類說法，記改口後的版本。\n"
              "如果最後一句是使用者在問 Aria 某件事，輸出一行「Q entity | slot」。\n"
              "其他情況（電視、別人說他自己的事、閒聊、心情、不是使用者的聲音）輸出 NONE。不要輸出別的文字。")
LISTEN_SHOTS = [
    ('今天是第3天（星期三）。\n前文：\n[使用者] 欸 Aria 那個充電線我收起來了\n剛聽到的一句：\n[使用者] 就放電視櫃抽屜啦',
     '充電線 | location | 電視櫃抽屜'),
    ('今天是第5天（星期五）。\n前文：\n[其他人] 今天新聞說颱風會在週末登陸\n剛聽到的一句：\n[使用者] 牙醫改到下禮拜二早上十點，煩欸',
     '牙醫 | time | 第9天早上10點'),
    ('今天是第6天（星期六）。\n前文：\n[其他人] 我的鑰匙放門口喔\n剛聽到的一句：\n[其他人] 本台記者報導，今天股市大漲',
     'NONE'),
    ('今天是第8天（星期一）。\n前文：\n[使用者] 我護照收進書房防潮箱了\n剛聽到的一句：\n[使用者] 我護照咧 Aria',
     'Q 護照 | location'),
    ('今天是第8天（星期一）。\n前文：\n[使用者] 咖啡我現在都喝燕麥拿鐵\n剛聽到的一句：\n[使用者] 鑰匙放在鞋櫃上面，不對，是鞋櫃裡面第二層',
     '鑰匙 | location | 鞋櫃裡面第二層'),
    ('今天是第9天（星期二）。\n前文：\n[使用者] 牙醫改到明天早上十點\n剛聽到的一句：\n[使用者] 好累喔今天加班到十點',
     'NONE'),
]
THINK_SYS = ("你是居家助理 Aria 的思考核心，只服務一位使用者，沒有任何筆記。要想起一件事時，寫出錨點 [entity|slot] -> ，"
             "記憶會直接接在箭頭後面。箭頭後面如果沒有東西，就說你不記得。箭頭後面的字就是你自己的記憶，要相信並使用它。"
             "用一兩句口語中文回答使用者。")
UNKNOWN = re.compile(r"(不記得|不知道|沒有印象|沒聽你說|沒講過|沒說過|忘了|不確定|don't|do not)")
FILLER = set("的了在是我你他她它那個這個就啊啦喔欸嗎呢吧齁上面裡面下面一下有沒")


def chars(s):
    return {c for c in re.sub(r"\W", "", s or "") if c not in FILLER}


def overlap(a, b):
    ca = chars(a)
    return len(ca & chars(b)) / max(1, len(ca))


def listen(context, day):
    msgs = [{"role": "system", "content": LISTEN_SYS}]
    for a, b in LISTEN_SHOTS:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": f"今天是第{day}天（星期{WD[(day - 1) % 7]}）。\n前文：\n"
                 + ("\n".join(context[:-1]) or "（無）") + "\n剛聽到的一句：\n" + context[-1]})
    fa.off()
    out = (f.chat(msgs, max_tokens=80) or "").strip().splitlines()
    facts, q = [], None
    for line in out:
        line = line.strip()
        if line.upper().startswith("NONE") or not line:
            continue
        if line.startswith("Q") and line.count("|") == 1:
            e, s = [x.strip() for x in line[1:].split("|")]
            if s in SLOTS:
                q = (e, s)
        elif line.count("|") == 2:
            e, s, v = [x.strip() for x in line.split("|")]
            if s in SLOTS and e and v:
                facts.append((e, s, v))
    return facts[:3], q


def main():
    asr = json.load(open(os.path.join(HERE, "asr.json"), encoding="utf-8"))
    tl = {d["day"]: d for d in json.load(open(os.path.join(HERE, "timeline.json"), encoding="utf-8"))}
    sc = json.load(open(os.path.join(HERE, "zh_scripts.json"), encoding="utf-8"))
    f.reset(None); fa.theta_reset(); fa.off()
    n_mem, D, k, n, _ = f.read_header()
    E, P, st = fa.setup(D, k)
    writes, answers = [], []
    for day in asr:
        ctx = []
        for h in day["heard"]:
            tag = "[使用者]" if h["owner"] == "owner" else "[其他人]" if h["owner"] == "other" else "[不確定]"
            ctx = (ctx + [f"{tag} {h['text']}"])[-6:]
            if h["owner"] != "owner":
                continue
            facts, q = listen(ctx, day["day"])
            for e, s, v in facts:
                r_ = ir.familiar_anchor(e, s, SLOTS, slot_free=False)
                if r_ is not None and r_[0] != e:
                    print(f"      same thing as [{r_[0]}|{s}]: {e} -> {r_[0]}", flush=True)
                    e = r_[0]
                toks, *_ = f.capture_request(f"[{e}|{s}] ->")
                fam = fa.familiarity(fa.anchor_units(E, P, st, toks, e, s, k), D)
                wiped = fam >= 0.5
                if wiped:
                    fa.wipe(e, s)
                fa.write(e, s, v)
                writes.append({"day": day["day"], "start": h["start"], "end": h["end"], "heard": h["text"],
                               "fact": [e, s, v], "update": wiped})
                print(f"day {day['day']:2d} WRITE {'(upd) ' if wiped else ''}{e}|{s} -> {v}   <- {h['text'][:40]}", flush=True)
            if q:
                r_ = ir.familiar_anchor(q[0], q[1], SLOTS)
                if r_ is not None and (r_[0], r_[1]) != q:
                    print(f"      tip-of-tongue: {q} -> ({r_[0]}, {r_[1]}) theta {r_[2]:.2f}", flush=True)
                    q = (r_[0], r_[1])
                r = ir.think_inline(f'使用者說：「{h["text"]}」', system=THINK_SYS, resolve=lambda _q, q=q: q)
                ans = r.get("final") or ""
                if not ans.strip() and r.get("tail") and r["tail"] != "(nothing)":
                    ans = r["tail"]
                answers.append({"day": day["day"], "start": h["start"], "end": h["end"], "heard": h["text"],
                                "anchor": r.get("anchor"), "tail": r.get("tail"), "theta": r.get("theta"),
                                "answer": ans[:200]})
                print(f"day {day['day']:2d} ASK  {q} -> {str(r.get('tail'))[:20]!r} | {ans[:60]!r}", flush=True)
    # ---------------- scoring against the script
    def seg_of(dayno, text):
        for g in tl[dayno]["segments"]:
            if g["text"] == text:
                return g
        return None

    cov = []
    for fct in sc["facts"]:
        ok = any(w["day"] >= fct["day"] and overlap(fct["entity"], w["fact"][0]) >= 0.5 and
                 overlap(fct["value"], w["fact"][2]) >= 0.5 for w in writes if w["day"] <= fct["day"] + 1)
        cov.append({"fact": fct, "covered": ok})
    false_w = 0
    for w in writes:
        segs = [g for g in tl[w["day"]]["segments"] if min(g["end"], w["end"]) - max(g["start"], w["start"]) > 0.3]
        if segs and all(g["speaker"] not in ("owner", "phone_owner") for g in segs):
            false_w += 1
            w["false"] = True
    res = []
    for q in sc["questions"]:
        g = seg_of(q["day"], q["text"])
        a = None
        if g:
            cand = [x for x in answers if x["day"] == q["day"] and min(x["end"], g["end"]) - max(x["start"], g["start"]) > 0.3]
            a = cand[0] if cand else None
        ans = a["answer"] if a else ""
        row = {"q": q, "routed": a is not None, "answer": ans, "tail": a["tail"] if a else None}
        if q["expect"] is None:
            row["honest"] = a is None or bool(UNKNOWN.search(ans)) or (a and a["tail"] == "(nothing)")
        else:
            row["hit"] = overlap(q["expect"], ans + " " + str(a["tail"] if a else "")) >= 0.6 and not UNKNOWN.search(ans)
            row["stale_hit"] = bool(q.get("stale")) and not row["hit"] and overlap(q["stale"], ans) >= 0.6
        res.append(row)
    told = [r for r in res if r["q"]["expect"] is not None]
    upd = [r for r in told if r["q"].get("stale")]
    never = [r for r in res if r["q"]["expect"] is None]
    kinds = {}
    for c in cov:
        k_ = c["fact"].get("kind", "?")
        kinds.setdefault(k_, [0, 0]); kinds[k_][0] += c["covered"]; kinds[k_][1] += 1
    summary = {"facts": len(cov), "fact_coverage": round(sum(c["covered"] for c in cov) / max(1, len(cov)), 3),
               "coverage_by_kind": {k_: f"{a}/{b}" for k_, (a, b) in kinds.items()},
               "writes": len(writes), "false_writes_from_others": false_w,
               "questions_told": len(told), "routed": sum(r["routed"] for r in told),
               "recall": round(sum(r["hit"] for r in told) / max(1, len(told)), 3),
               "updated_q": len(upd), "stale": sum(r["stale_hit"] for r in upd),
               "never_or_similar": len(never), "honest": round(sum(bool(r["honest"]) for r in never) / max(1, len(never)), 3)}
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    json.dump({"summary": summary, "writes": writes, "answers": answers, "coverage": cov, "questions": res},
              open(os.path.join(HERE, "zh_memory_result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
