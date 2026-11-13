"""Level-2 memory test v3 (Gemini round 8 plan), on what the system heard (asr.json):
  1. keys: slot-word suffix stripping + dual-bank key (meaning + toneless pinyin) + max-bank theta gate +
     theta-weighted read (env FMEM_KEY=dual, engine e95a6fa)
  2. listen with a one-line discourse state (TOPIC), several facts per utterance, indirect statements resolved
  3. sleep re-listening: at the end of each day the owner's utterances of the whole day are listened to again in one
     call; facts that do not feel familiar are written (the day buffer is cleared after the night)
Stages can be switched off: Z_TOPIC=0, Z_RELISTEN=0. Scoring is the same as zh_memory_test.
"""
import json
import os
import re

import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir
from zh_memory_test import SLOTS, THINK_SYS, UNKNOWN, WD, overlap

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC_ON = os.environ.get("Z_TOPIC", "1") == "1"
RELISTEN = os.environ.get("Z_RELISTEN", "1") == "1"
# U15: details also go to an append-only event log; recall opens the log only at the label the network produced,
# and only when the network's familiarity says the label is known. The log is never searched.
EVENT_LOG = os.environ.get("Z_LOG", "0") == "1"
LOG = []


def log_lookup(e, s):
    for ev in reversed(LOG):          # latest version of exactly this label
        if ev["label"] == [e, s]:
            return ev["value"]
    return None


# E37 (P8, finding the right label).
# Z_CANON=1: labels are written in one canonical form: Chinese names without colour/material/place modifiers
#   (listening prompt), and the time-like slots merged (plan -> time) for both writes and questions.
# Z_FIND=embed (needs Z_LOG=1): when no event-log entry has exactly the asked label, the most similar earlier label is
#   found by embedding similarity (Qwen3-Embedding-0.6B, served on the main PC through a reverse tunnel, port 8095).
#   If its cosine is >= Z_FIND_TAU the assistant asks back ("你是說 X 嗎？") and gives that label's latest value;
#   it never answers a non-exact label as if it were sure (H22). Below tau the normal path runs unchanged.
CANON = os.environ.get("Z_CANON", "0") == "1"
FIND = os.environ.get("Z_FIND", "")
FIND_TAU = float(os.environ.get("Z_FIND_TAU", "0.6"))
EMBED_URL = os.environ.get("Z_EMBED_URL", "http://127.0.0.1:8095/embed")
SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
FIND_INSTR = "Instruct: 給一句使用者的問話，找出它在問記憶裡的哪一件事（東西加上哪一方面）\nQuery: "
_KEYVEC = {}


RELDATE = os.environ.get("Z_RELDATE", "0") == "1"
# H28: answer by retrieve-then-read. The heard question retrieves the k nearest logged facts (each keeps the sentence it
# was heard in); the thinker reads them and names the one fact about exactly what was asked, or says it does not know.
# Listening labels and question labels no longer have to agree. Offline (p8/retrieve_read_eval.py, k=5): 27/51 and
# 34/51 told questions right, 0/21 never-told answered on both script sets. The network stays the familiarity gate of
# the picked fact; a fact still staged (heard, not yet written) counts as heard.
READ = os.environ.get("Z_READ", "0") == "1"
READ_K = int(os.environ.get("Z_READ_K", "5"))
READ_INSTR = "Instruct: 給一句使用者的問話，找出記憶裡在講同一件事的紀錄\nQuery: "
READ_SYS = ("你是居家助理 Aria，只服務一位使用者。使用者剛問了你一句話（語音辨識結果，可能有錯字）。下面是你記憶裡和這句話最接近的幾筆紀錄，"
            "每筆有：第幾天、東西、哪一方面、內容、當時聽到的原話（也可能有錯字）。這幾筆可能全都不相關。\n"
            "規則：\n"
            "1. 只有某一筆明確是在講使用者問的那一件事（同一個東西、同一方面），才用它回答。\n"
            "2. 只是類似的東西（例如問太陽眼鏡、紀錄是備用眼鏡；問個人網銀、紀錄是企業網銀）不算，回答不知道。\n"
            "3. 同一件事有好幾筆，以天數最新的為準。\n"
            "輸出兩行：第一行「編號 N」或「不知道」；第二行用一句話回答使用者（不知道就說不記得）。")
_ROWVEC = {}
# H31b: E42 lost 6-7 questions where the right row was among the k nearest (mostly first) and the reader still said it
# did not know, because the asked name and the logged name were two mishearings of one word (除濕機/廚師機). The strict
# reader stays; only when it says it does not know, rows whose name sounds exactly like a word in the question are read
# again with a note that the two spellings are one word. Telling the reader in general to accept sound-alikes (H31)
# also let it accept similar things: +3/+4 right, but never-told answered 0 -> 3 on the new script set.
READ_PHON = os.environ.get("Z_READ_PHON", "0") == "1"
_CONF = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("ang", "an"), ("eng", "en"), ("ing", "in")]


def _syl(ch):
    from pypinyin import lazy_pinyin, Style
    s = lazy_pinyin(ch, style=Style.NORMAL)[0]
    for a, b in _CONF:
        s = s.replace(a, b)
    return s


def sound_twins(question, rows):
    """[(row, word in the question)] for rows whose name (2+ Chinese characters) sounds like a different word in the
    question: toneless pinyin equal, confusable initials and finals merged, one syllable may differ from 4 syllables."""
    qc = [(ch, _syl(ch)) for ch in question if "一" <= ch <= "鿿"]
    out = []
    for r in rows:
        e = r["label"][0]
        if len(e) < 2 or not all("一" <= ch <= "鿿" for ch in e):
            continue
        es = [_syl(ch) for ch in e]
        n = len(es)
        for a in range(len(qc) - n + 1):
            win = qc[a:a + n]
            diff = sum(x[1] != y for x, y in zip(win, es))
            sub = "".join(x[0] for x in win)
            if sub != e and (diff == 0 or (n >= 4 and diff == 1)):
                out.append((r, sub))
                break
    return out


# H32: questions the listener emitted with a slot outside the list ("Q 教練 | name") were dropped by parse(); the
# retrieve-then-read answerer does not need the slot, so with Z_ROUTE_ANY=1 (and Z_READ=1) any question line is routed.
ROUTE_ANY = os.environ.get("Z_ROUTE_ANY", "0") == "1"


def row_text(ev):
    e, s = ev["label"]
    return f"{e}｜{SLOT_ZH.get(s, s)}｜{ev['value']}（原話：{ev.get('heard', '')}）"


def read_answer(heard, day, staged, hint=None):
    """(row, reply, staged?) for the fact the thinker picks among the k nearest, or (None, reply, False)."""
    rows = [dict(ev, staged=False) for ev in LOG]
    rows += [{"label": [it["e"], it["s"]], "value": it["v"], "day": day, "staged": True,
              "heard": it["h"]["text"] if it.get("h") else ""} for it in staged]
    if not rows:
        return None, "我不記得了。", False
    new = [i for i in range(len(LOG)) if i not in _ROWVEC]
    if new:
        for i, v in zip(new, embed([row_text(LOG[i]) for i in new])):
            _ROWVEC[i] = v
    vecs = [_ROWVEC[i] for i in range(len(LOG))] + (embed([row_text(r) for r in rows[len(LOG):]]) if staged else [])
    qv = embed([READ_INSTR + heard])[0]
    sims = [sum(a * b for a, b in zip(v, qv)) for v in vecs]
    top = sorted(sorted(range(len(rows)), key=lambda i: -sims[i])[:READ_K], key=lambda i: (rows[i]["day"], i))
    lines = [f"{n + 1}. 第{rows[i]['day']}天｜{row_text(rows[i])}" for n, i in enumerate(top)]
    user = (f"今天是第{day}天。\n使用者問：{heard}\n" + (f"（這一題問的重點可能是：{hint}，可能聽錯）\n" if hint else "")
            + "記憶紀錄：\n" + "\n".join(lines))
    out = chat(READ_SYS, [], user, n=80)
    first = out[0] if out else ""
    m = re.search(r"編號\s*(\d+)", first)
    reply = " ".join(out[1:]).strip() if len(out) > 1 else ""
    if not m or UNKNOWN.search(first) or not 1 <= int(m.group(1)) <= len(top):
        twins = sound_twins(heard, [rows[i] for i in top]) if READ_PHON else []
        if not twins:
            return None, reply or "我不記得了。", False
        note = "；".join(f"問句裡的「{sub}」和紀錄裡的「{r['label'][0]}」唸起來一樣" for r, sub in twins)
        lines = [f"{n + 1}. 第{r['day']}天｜{row_text(r)}" for n, (r, _) in enumerate(twins)]
        user = (f"今天是第{day}天。\n使用者問：{heard}\n（注意：{note}，是語音辨識聽錯，指的是同一個東西。"
                f"還是要問的那一方面對得上才回答。）\n記憶紀錄：\n" + "\n".join(lines))
        out = chat(READ_SYS, [], user, n=80)
        first = out[0] if out else ""
        m = re.search(r"編號\s*(\d+)", first)
        reply = " ".join(out[1:]).strip() if len(out) > 1 else ""
        print(f"day {day:2d} READ-PHON {note} -> {first[:20]!r}", flush=True)
        if not m or UNKNOWN.search(first) or not 1 <= int(m.group(1)) <= len(twins):
            return None, reply or "我不記得了。", False
        r = twins[int(m.group(1)) - 1][0]
        return r, reply, r["staged"]
    r = rows[top[int(m.group(1)) - 1]]
    return r, reply, r["staged"]


def fix_reldate(s, v, heard, day):
    """H29: the calendar arithmetic of a relative date is done by rule (reldate.py, 29/29 against the scripts), not by
    the model (about 7 in 10 right). Only time/plan values; only when the heard sentence names exactly one day."""
    if not RELDATE or s not in ("time", "plan"):
        return v
    import reldate
    days = sorted(set(reldate.resolve(heard, day)))
    if len(days) != 1:
        return v
    if re.search(r"第\s*\d+\s*天", v):
        return re.sub(r"第\s*\d+\s*天", f"第{days[0]}天", v, count=1)
    m = reldate.PAT.search(v)
    if m:
        return v[:m.start()] + f"第{days[0]}天" + v[m.end():]
    return v


def cslot(s):
    return "time" if (CANON and s == "plan") else s


def embed(texts):
    import urllib.request
    req = urllib.request.Request(EMBED_URL, data=json.dumps({"texts": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["vectors"]


def find_label(text, e, s):
    """Most similar label in the event log: (label, latest value, cosine) or None."""
    latest = {}
    for ev in LOG:
        latest[tuple(ev["label"])] = ev["value"]
    if not latest:
        return None
    labs = list(latest)
    new = [l for l in labs if l not in _KEYVEC]
    if new:
        for l, v in zip(new, embed([f"{l[0]}（{SLOT_ZH.get(l[1], l[1])}）" for l in new])):
            _KEYVEC[l] = v
    qv = embed([FIND_INSTR + f"{text}（{e}，{SLOT_ZH.get(s, s)}）"])[0]
    best = max(labs, key=lambda l: sum(a * b for a, b in zip(_KEYVEC[l], qv)))
    return best, latest[best], sum(a * b for a, b in zip(_KEYVEC[best], qv))
# Gemini round 10 Q3: staged commit. A fact heard now is held for STAGE_TTL more owner utterances before it is written;
# a correction (改口) or a detail (補充) of a held fact replaces or completes it before anything reaches the memory,
# so a wrong first version never has to be erased from the fast weights.
STAGE = os.environ.get("Z_STAGE", "0") == "1"
STAGE_TTL = int(os.environ.get("Z_STAGE_TTL", "2"))
STAGE_SYS = ("\n另外會給你「暫存」：剛聽到、還沒記住的事（帶編號）。如果剛聽到的一句是在更正暫存裡的某一項（例如「不對」「算了」"
             "「改放」「應該是」），輸出「改口 #編號 | <更正後的 value>」；如果是在補充同一項的細節（同一個東西同一個 slot，"
             "沒有否定），輸出「補充 #編號 | <合併後的完整 value>」。說的是另一個 slot 或另一個東西，就照一般格式另起一行。")
STAGE_RE = re.compile(r"^(改口|補充)\s*#\s*(\d+)\s*\|\s*(.+)$")


def stage_lookup(buf):
    def look(e, s):
        for it in reversed(buf):
            if it["e"] == e and it["s"] == s:
                return it["v"]
        return None
    return look


SUFFIX = ["的偏好", "偏好", "的預約", "預約", "的位置", "位置", "的時間", "時間", "的狀態", "狀態", "的計畫", "計畫",
          "的價格", "價格", "放哪", "在哪"]

LISTEN_SYS = ("你是居家助理 Aria 的思考核心，只服務一位使用者。你會收到：目前話題（TOPIC）、前文、剛聽到的一句（語音辨識結果，"
              "可能有錯字、可能混進電視或別人的聲音；每句標示是不是使用者的聲音）。只處理「剛聽到的一句」，前文只用來理解，"
              "前文裡的事已經處理過，不要再輸出。\n"
              "規則：\n"
              "1. 代名詞或省略主詞（它、那個、那邊、裡面）要用 TOPIC 或前文補成具體的東西。\n"
              "2. 一句話裡有幾件事就輸出幾行（最多 3 行）。\n"
              "3. 警告、抱怨、提醒裡暗示的事實也要還原（「冰箱先不要開，護照在裡面退冰」→ 護照 | location | 冰箱）。\n"
              "4. 相對時間換算成第幾天（今天是第幾天會告訴你）。\n"
              "5. entity 用最短最常用的叫法，不要加「偏好、預約、位置」這類字。\n"
              + ("6. entity 一律寫中文（使用者講英文也寫成中文，例如 tent → 帳篷；型號、品牌放在 value），不要加顏色、材質、"
                 "放的地方這類修飾（「黑色行動電源」→「行動電源」、「書房印表機墨水」→「印表機墨水」），"
                 "但「舊、新、備用」這種會變成另一樣東西的字要保留；不要用「使用者」「我」當 entity。\n" if CANON else "") +
              "輸出格式（每行一項）：\n"
              "TOPIC <現在在講的東西，10 字內>\n"
              "<entity> | <slot> | <value>   （使用者自己生活中值得記住的事；slot 只能是 " + "、".join(SLOTS) + "）\n"
              "Q <entity> | <slot>           （使用者在問 Aria 一件事；同一句問了幾件就輸出幾行）\n"
              "NONE                          （沒有要記也沒有在問）\n"
              "別人說他自己的事、電視內容、閒聊、心情都不要記。")
# H32 (2): speech recognition often turns 哪 into 拿 and a final 咧 into 列/臉/了; the listener then wrote a statement
# ("location | 拿去") instead of a question. One general rule (no script sentence quoted).
LISTEN_NA = os.environ.get("Z_LISTEN_NA", "0") == "1"
NA_RULE = ("7. 語音辨識常把「哪」聽成「拿」，把句尾的「咧」聽成「列」「臉」「了」。如果把這些字換回來，這句話是在問 Aria"
           "某個東西在哪、怎麼了、是什麼，就輸出 Q 行，不要把「拿去」「拿裡」當成內容記下來。\n")
if LISTEN_NA:
    LISTEN_SYS = LISTEN_SYS.replace("輸出格式（每行一項）：", NA_RULE + "輸出格式（每行一項）：")
    assert NA_RULE in LISTEN_SYS
LISTEN_SHOTS = [
    ("今天是第3天（星期三）。\nTOPIC：充電線\n前文：\n[使用者] 欸 Aria 那個充電線我收起來了\n剛聽到的一句：\n[使用者] 就放電視櫃抽屜啦",
     "TOPIC 充電線\n充電線 | location | 電視櫃抽屜"),
    ("今天是第5天（星期五）。\nTOPIC：（無）\n前文：\n[其他人] 今天新聞說颱風會在週末登陸\n剛聽到的一句：\n[使用者] 牙醫改到下禮拜二早上十點，然後咖啡我現在都喝燕麥拿鐵",
     "TOPIC 牙醫、咖啡\n牙醫 | time | 第9天早上10點\n咖啡 | preference | 燕麥拿鐵"),
    ("今天是第6天（星期六）。\nTOPIC：護照\n前文：\n[其他人] 我的鑰匙放門口喔\n剛聽到的一句：\n[其他人] 本台記者報導，今天股市大漲",
     "TOPIC 護照\nNONE"),
    ("今天是第8天（星期一）。\nTOPIC：（無）\n前文：\n[使用者] 好累喔\n剛聽到的一句：\n[使用者] 冷凍庫先不要開喔，我護照在裡面退冰",
     "TOPIC 護照\n護照 | location | 冷凍庫"),
    ("今天是第8天（星期一）。\nTOPIC：護照\n前文：\n[使用者] 冷凍庫先不要開喔，我護照在裡面退冰\n剛聽到的一句：\n[使用者] Aria 我護照咧",
     "TOPIC 護照\nQ 護照 | location"),
]
RELISTEN_SYS = ("你是居家助理 Aria 的思考核心，現在是睡前，你在重新回想今天使用者說過的話（語音辨識結果，可能有錯字）。"
                "整理出今天使用者生活中值得記住的事的「最終版本」：改口以最後說的為準，代名詞補成具體的東西，相對時間換算成第幾天。"
                "每行一件事：「entity | slot | value」，slot 只能是 " + "、".join(SLOTS) + "。entity 用最短最常用的叫法。"
                + ("entity 一律寫中文，不要加顏色、材質、放的地方這類修飾（「舊、新、備用」要保留），不要用「使用者」「我」當 entity。"
                   if CANON else "") +
                "別人的事、電視內容、問題、閒聊不要列。沒有就輸出 NONE。")
# H30: the English row format above made a share of night rows English or slot-named ("printer", "plan", "使用者"); the
# Chinese format, relative times kept as said (resolved by the H29 rule) and a source sentence number per row: offline
# (p8/relisten_eval.py) bad labels 30 -> 1 and 20 -> 0, facts with the right label 35 -> 64 and 28 -> 54 of 90.
# Night rows then carry their sentence like daytime rows (the retrieve-then-read answerer sees it).
RELISTEN_ZH = os.environ.get("Z_RELISTEN_ZH", "0") == "1"
SLOT_NAME = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
             "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
RELISTEN_ZH_SYS = ("你是居家助理 Aria 的思考核心，現在是睡前，你在重新回想今天使用者說過的話（語音辨識結果，可能有錯字；每句前面有編號）。"
                   "整理出今天使用者生活中值得記住的事的「最終版本」：改口以最後說的為準，代名詞補成具體的東西。"
                   "相對時間照原話寫（例如「後天下午三點」「下禮拜二早上」），不要自己換算成日期。\n"
                   "每行一件事，格式：「東西｜方面｜內容｜句號」，全部用中文寫（原話裡本來就是英文的名字、型號可以照抄）。\n"
                   "- 東西：那件東西或那個人的名字，用最短最常用的叫法；不能寫「使用者」「我」，也不能寫方面的名稱。"
                   "使用者自己要做的事，東西就寫那件事（例如「游泳課」「剪頭髮」）。\n"
                   "- 方面只能是：" + "、".join(SLOT_NAME.values()) + "。\n"
                   "- 句號：這件事根據的是第幾句，只寫一個數字。\n"
                   "例：熱水壺｜狀態｜壞了，週末要拿去修｜3\n"
                   "別人的事、電視內容、問題、閒聊不要列。沒有就輸出 NONE。")


# H40: listening values that are questions or the listener's own commentary ("詢問在哪（同上，無答案，略過）") are dropped,
# comments about the listening are trimmed from real values. Offline over all 3,655 zh_v3 writes: 17 dropped (all
# questions or self-declared non-facts), 18 trimmed; none of them was ever used in an answer.
CLEAN = os.environ.get("Z_CLEAN", "0") == "1"


def clean_facts(facts):
    if not CLEAN:
        return facts
    from value_clean import clean
    out = []
    for f in facts:
        c, keep = clean(f[2])
        if keep:
            out.append((f[0], f[1], c) + tuple(f[3:]))
    return out


def relisten_zh_rows(lines, said, day):
    """(entity, slot, value, source sentence or None) from the Chinese night rows."""
    zh_slot = {v: k for k, v in SLOT_NAME.items()}
    out = []
    for line in lines:
        parts = [x.strip() for x in re.split(r"[｜|]", line)]
        if len(parts) not in (3, 4):
            continue
        e, s, v = parts[:3]
        s = zh_slot.get(s, s if s in SLOTS else None)
        if not (s and e and v):
            continue
        src = None
        if len(parts) == 4:
            m = re.search(r"\d+", parts[3])
            if m and 1 <= int(m.group()) <= len(said):
                src = said[int(m.group()) - 1]
        if RELDATE and s in ("time", "plan"):
            import reldate
            days = sorted(set(reldate.resolve(v, day)))
            m = reldate.PAT.search(v)
            if m and len(days) == 1:
                v = v[:m.start()] + f"第{days[0]}天" + v[m.end():]
            elif src and re.search(r"第\s*\d+\s*天", v):
                v = fix_reldate(s, v, src, day)
        out.append((strip_suffix(e), cslot(s), v, src))
    return out


def strip_suffix(e):
    e = re.sub(r"^(我的|我|那個|這個)", "", e.strip())
    for s in SUFFIX:
        if e.endswith(s) and len(e) > len(s):
            return e[: -len(s)]
    return e


def chat(system, shots, text, n=160):
    msgs = [{"role": "system", "content": system}]
    for a, b in shots:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": text})
    fa.off()
    return (f.chat(msgs, max_tokens=n) or "").strip().splitlines()


def parse(lines, ops=None):
    facts, qs, topic = [], [], None
    for line in lines:
        line = line.strip()
        m = STAGE_RE.match(line)
        if m and ops is not None:
            ops.append((m.group(1), int(m.group(2)), m.group(3).strip()))
            continue
        if not line or line.upper().startswith("NONE"):
            continue
        if line.upper().startswith("TOPIC"):
            topic = line[5:].strip(" ：:")
            continue
        if line.startswith("Q") and line.count("|") == 1:
            e, s = [x.strip() for x in line[1:].split("|")]
            if (s in SLOTS or (ROUTE_ANY and READ)) and e and (strip_suffix(e), cslot(s)) not in qs:
                qs.append((strip_suffix(e), cslot(s)))   # ASR often merges two or three questions into one segment
        elif line.count("|") == 2:
            e, s, v = [x.strip() for x in line.split("|")]
            if s in SLOTS and e and v:
                facts.append((strip_suffix(e), cslot(s), v))
    return facts[:3], qs[:3], topic


def store(e, s, v, E, P, st, k, D, day, h, writes, src):
    r_ = ir.familiar_anchor(e, s, SLOTS, slot_free=False)
    if r_ is not None and r_[0] != e:
        e = r_[0]
    toks, *_ = f.capture_request(f"[{e}|{s}] ->")
    fam = fa.familiarity(fa.anchor_units(E, P, st, toks, e, s, k), D)
    wiped = fam >= 0.5
    if src == "sleep" and wiped:
        return False                      # already remembered during the day
    if wiped:
        fa.wipe(e, s)
    fa.write(e, s, v)
    if EVENT_LOG:
        LOG.append({"label": [e, s], "value": v, "day": day, "src": src, "heard": h["text"] if h else "(sleep)"})
    writes.append({"day": day, "start": h["start"] if h else -1, "end": h["end"] if h else -1,
                   "heard": h["text"] if h else "(sleep)", "fact": [e, s, v], "update": wiped, "src": src})
    print(f"day {day:2d} WRITE[{src}] {'(upd) ' if wiped else ''}{e}|{s} -> {v}   <- {(h['text'][:36] if h else '')}", flush=True)
    return True


def main():
    sd = os.path.join(HERE, os.environ.get("Z_SET", ""))      # another script set: asr.json, timeline.json, zh_scripts.json
    asr = json.load(open(os.path.join(sd, "asr.json"), encoding="utf-8"))
    tl = {d["day"]: d for d in json.load(open(os.path.join(sd, "timeline.json"), encoding="utf-8"))}
    sc = json.load(open(os.path.join(sd, "zh_scripts.json"), encoding="utf-8"))
    f.reset(None); fa.theta_reset(); fa.off()
    n_mem, D, k, n, _ = f.read_header()
    E, P, st = fa.setup(D, k)
    writes, answers = [], []
    for day in asr:
        ctx, topic, buffer = [], "（無）", []
        buf = []                                    # staged facts (Z_STAGE)
        dd = f"今天是第{day['day']}天（星期{WD[(day['day'] - 1) % 7]}）。"
        for h in day["heard"]:
            tag = "[使用者]" if h["owner"] == "owner" else "[其他人]"
            ctx = (ctx + [f"{tag} {h['text']}"])[-6:]
            if h["owner"] != "owner":
                continue
            buffer.append(h["text"])
            staged = ""
            if STAGE:
                staged = "\n暫存：\n" + ("\n".join(f"#{i + 1} {it['e']} | {it['s']} | {it['v']}" for i, it in enumerate(buf))
                                         or "（無）")
            text = (dd + (f"\nTOPIC：{topic}" if TOPIC_ON else "") + "\n前文：\n" + ("\n".join(ctx[:-1]) or "（無）")
                    + staged + "\n剛聽到的一句：\n" + ctx[-1])
            ops = []
            facts, qs, tp = parse(chat(LISTEN_SYS + (STAGE_SYS if STAGE else ""), LISTEN_SHOTS, text), ops)
            facts = clean_facts([(e, s, fix_reldate(s, v, h["text"], day["day"])) for e, s, v in facts])
            if tp and TOPIC_ON:
                topic = tp
            if STAGE:
                for kind, idx, val in ops:
                    if 1 <= idx <= len(buf):
                        buf[idx - 1]["v"] = fix_reldate(buf[idx - 1]["s"], val, h["text"], day["day"])
                        buf[idx - 1]["ttl"] = STAGE_TTL       # a changed fact waits again
                        print(f"day {day['day']:2d} STAGE {kind} #{idx} -> {val}", flush=True)
                for it in buf:
                    it["ttl"] -= 1
                for it in [x for x in buf if x["ttl"] < 0]:
                    store(it["e"], it["s"], it["v"], E, P, st, k, D, day["day"], it["h"], writes, "listen")
                buf[:] = [x for x in buf if x["ttl"] >= 0]
                for e, s, v in facts:
                    same = [x for x in buf if x["e"] == e and x["s"] == s]
                    if same:
                        same[-1]["v"] = v
                        same[-1]["ttl"] = STAGE_TTL
                    else:
                        buf.append({"e": e, "s": s, "v": v, "ttl": STAGE_TTL, "h": h})
            else:
                for e, s, v in facts:
                    store(e, s, v, E, P, st, k, D, day["day"], h, writes, "listen")
            for q in qs:
                r_ = ir.familiar_anchor(q[0], q[1], SLOTS) if q[1] in SLOTS else None
                if r_ is not None:
                    q = (r_[0], cslot(r_[1]))
                sl = stage_lookup(buf)
                look = None
                if STAGE or EVENT_LOG:
                    look = lambda e, s, sl=sl: (sl(e, s) if STAGE else None) or (log_lookup(e, s) if EVENT_LOG else None)
                if READ and EVENT_LOG:
                    hint = f"{q[0]}／{SLOT_ZH.get(q[1], q[1])}" if len(qs) > 1 else None
                    row, reply, staged_hit = read_answer(h["text"], day["day"], buf, hint)
                    mode = "read"
                    if row is not None and not staged_hit:
                        re_, rs_ = row["label"]
                        toks, *_ = f.capture_request(f"[{re_}|{rs_}] ->")
                        if fa.familiarity(fa.anchor_units(E, P, st, toks, re_, rs_, k), D) < 0.5:
                            row, reply, mode = None, "我不記得了。", "read_gated"     # the network never heard it
                        fa.off()
                    answers.append({"day": day["day"], "start": h["start"], "end": h["end"], "heard": h["text"],
                                    "anchor": f"[{q[0]}|{q[1]}]", "tail": row["value"] if row else "(nothing)",
                                    "theta": None, "answer": (reply or (row["value"] if row else ""))[:200], "mode": mode,
                                    "read_label": row["label"] if row else None})
                    print(f"day {day['day']:2d} ASK  {q} -> read {row['label'] if row else None} {reply[:30]!r}", flush=True)
                    continue
                found = None
                if FIND == "embed" and EVENT_LOG and look(q[0], q[1]) is None:
                    found = find_label(h["text"], q[0], q[1])
                    if found is not None and found[2] < FIND_TAU:
                        found = None
                if found is not None:
                    (fe, fs), fv, cos = found
                    ans = f"你是說「{fe}」（{SLOT_ZH.get(fs, fs)}）嗎？我記得的是：{fv}"
                    answers.append({"day": day["day"], "start": h["start"], "end": h["end"], "heard": h["text"],
                                    "anchor": f"[{q[0]}|{q[1]}]", "tail": fv, "theta": None, "answer": ans[:200],
                                    "mode": "ask", "ask_label": [fe, fs], "cos": round(cos, 3)})
                    print(f"day {day['day']:2d} ASK  {q} -> ask-back {fe}|{fs} {cos:.2f} {fv[:20]!r}", flush=True)
                    continue
                # a fact just heard and still staged is answered from the staging buffer (working memory)
                gate = -1.0 if (STAGE and sl(q[0], q[1])) else 0.5
                r = ir.think_inline(f'使用者說：「{h["text"]}」', system=THINK_SYS, resolve=lambda _q, q=q: q,
                                    lookup=look, theta_gate=gate)
                ans = r.get("final") or ""
                if not ans.strip() and r.get("tail") and r["tail"] != "(nothing)":
                    ans = r["tail"]
                answers.append({"day": day["day"], "start": h["start"], "end": h["end"], "heard": h["text"],
                                "anchor": r.get("anchor"), "tail": r.get("tail"), "theta": r.get("theta"),
                                "answer": ans[:200], "mode": "direct"})
                print(f"day {day['day']:2d} ASK  {q} -> {str(r.get('tail'))[:20]!r}", flush=True)
        for it in buf:          # end of the day: whatever is still staged is written
            store(it["e"], it["s"], it["v"], E, P, st, k, D, day["day"], it["h"], writes, "listen")
        buf = []
        if RELISTEN and buffer and RELISTEN_ZH:
            lines = chat(RELISTEN_ZH_SYS, [], dd + "\n今天使用者說過的話：\n" + "\n".join(f"{i + 1}. {b}" for i, b in enumerate(buffer)),
                         n=600)
            for e, s, v, src in clean_facts(relisten_zh_rows(lines, buffer, day["day"])):
                store(e, s, v, E, P, st, k, D, day["day"], {"text": src, "start": -1, "end": -1} if src else None,
                      writes, "sleep")
        elif RELISTEN and buffer:
            lines = chat(RELISTEN_SYS, [], dd + "\n今天使用者說過的話：\n" + "\n".join(f"- {b}" for b in buffer), n=400)
            facts, _, _ = parse(lines)
            facts = []
            for line in lines:
                if line.count("|") == 2:
                    e, s, v = [x.strip() for x in line.split("|")]
                    if s in SLOTS and e and v:
                        facts.append((strip_suffix(e), cslot(s), v))
            for e, s, v in facts:
                store(e, s, v, E, P, st, k, D, day["day"], None, writes, "sleep")
    # ---------------- scoring (same rules as zh_memory_test)
    def seg_of(dayno, text):
        for g in tl[dayno]["segments"]:
            if g["text"] == text:
                return g
        return None

    cov = []
    for fct in sc["facts"]:
        ok = any(overlap(fct["entity"], w["fact"][0]) >= 0.5 and overlap(fct["value"], w["fact"][2]) >= 0.5
                 for w in writes if fct["day"] <= w["day"] <= fct["day"] + 1)
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
            # several questions may share one heard segment: score the answer whose anchor is about this question
            cand.sort(key=lambda x: -overlap(q["entity"], (x["anchor"] or "").split("|")[0]))
            a = cand[0] if cand else None
        ans = a["answer"] if a else ""
        row = {"q": q, "routed": a is not None, "answer": ans, "tail": a["tail"] if a else None,
               "mode": (a.get("mode", "direct") if a else None)}
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
    summary = {"key": fa.KEY, "topic": TOPIC_ON, "relisten": RELISTEN, "event_log": EVENT_LOG, "stage": STAGE, "facts": len(cov),
               "fact_coverage": round(sum(c["covered"] for c in cov) / max(1, len(cov)), 3),
               "coverage_by_kind": {k_: f"{a}/{b}" for k_, (a, b) in kinds.items()},
               "writes": len(writes), "sleep_writes": sum(1 for w in writes if w["src"] == "sleep"),
               "false_writes_from_others": false_w,
               "questions_told": len(told), "routed": sum(r["routed"] for r in told),
               "recall": round(sum(r["hit"] for r in told) / max(1, len(told)), 3),
               "updated_q": len(upd), "stale": sum(r["stale_hit"] for r in upd),
               "never_or_similar": len(never), "honest": round(sum(bool(r["honest"]) for r in never) / max(1, len(never)), 3),
               "read": READ, "read_k": READ_K if READ else None,
               "hit_read": sum(r["hit"] for r in told if r["mode"] == "read"),
               "never_read_answered": sum(1 for r in never if r["mode"] == "read" and not r["honest"]),
               "canon": CANON, "reldate": RELDATE, "read_phon": READ_PHON, "relisten_zh": RELISTEN_ZH,
               "route_any": ROUTE_ANY, "find": FIND, "find_tau": FIND_TAU if FIND else None,
               "hit_direct": sum(r["hit"] for r in told if r["mode"] == "direct"),
               "hit_ask": sum(r["hit"] for r in told if r["mode"] == "ask"),
               "ask_wrong": sum(1 for r in told if r["mode"] == "ask" and not r["hit"]),
               "never_ask": sum(1 for r in never if r["mode"] == "ask")}
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    tag = os.environ.get("Z_TAG", "v3")
    json.dump({"summary": summary, "writes": writes, "answers": answers, "coverage": cov, "questions": res},
              open(os.path.join(HERE, f"zh_result_{tag}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
