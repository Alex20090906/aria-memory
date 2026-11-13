"""H28 offline: answer from the event log without labels. Every logged fact keeps the sentence it was heard in.
At question time the heard question (speech-recognised, as in the run) retrieves the k nearest logged facts by
Qwen3-Embedding similarity, and LLM server (normal service on the AI PC, port 8081) reads them and either names the one
fact that is about exactly what was asked, or says it does not know. Labels written at listening time and labels parsed
from the question no longer have to agree, which is where most losses were (P8).

Replays the writes of a finished zh_v3 run; scoring as rescore_norm (numbers and weekdays normalised).

    python retrieve_read_eval.py <zh_result_*.json> [...]      (P8_FM=<folder with timeline.json> for set2)
env: RR_K (default "3,5"), LLM_API_KEY
"""
import json
import os
import re
import sys
import time
import urllib.request

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import label_finder_eval as lf  # noqa: E402
import numpy as np  # noqa: E402
from label_finder_embed import SLOT_ZH  # noqa: E402
from rescore_norm import norm, UNKNOWN  # noqa: E402

URL = "http://<thinker-host>:8081/v1/chat/completions"
INSTR = "Instruct: 給一句使用者的問話，找出記憶裡在講同一件事的紀錄\nQuery: "
SYS = ("你是居家助理 Aria，只服務一位使用者。使用者剛問了你一句話（語音辨識結果，可能有錯字）。下面是你記憶裡和這句話最接近的幾筆紀錄，"
       "每筆有：第幾天、東西、哪一方面、內容、當時聽到的原話（也可能有錯字）。這幾筆可能全都不相關。\n"
       "規則：\n"
       "1. 只有某一筆明確是在講使用者問的那一件事（同一個東西、同一方面），才用它回答。\n"
       "2. 只是類似的東西（例如問太陽眼鏡、紀錄是備用眼鏡；問個人網銀、紀錄是企業網銀）不算，回答不知道。\n"
       "3. 同一件事有好幾筆，以天數最新的為準。\n"
       "輸出兩行：第一行「編號 N」或「不知道」；第二行用一句話回答使用者（不知道就說不記得）。")
# H31: the reader is told that names are often misheard as sound-alikes (E42: the right row was among the k nearest,
# often first, and the reader still said it did not know, mostly because the asked name and the logged name were two
# different mishearings of one word). Examples are words that occur in neither script set.
SYS_SOUND = ("你是居家助理 Aria，只服務一位使用者。使用者剛問了你一句話（語音辨識結果，常有同音或近音錯字）。下面是你記憶裡和這句話最接近的幾筆紀錄，"
             "每筆有：第幾天、東西、哪一方面、內容、當時聽到的原話（也常有錯字）。這幾筆可能全都不相關。\n"
             "規則：\n"
             "1. 東西的名字常被聽成讀音相近的字（例如「洗衣機」聽成「洗一機」、「王經理」聽成「王金理」、「熱水壺」聽成「熱水湖」）。"
             "問話裡的名字和紀錄裡的名字唸起來相近、而且問的方面對得上，就算同一件事，用它回答。\n"
             "2. 唸起來不像的另一個東西，即使是同一類（例如問太陽眼鏡、紀錄是備用眼鏡；問個人網銀、紀錄是企業網銀），不算，回答不知道。\n"
             "3. 同一件事有好幾筆，以天數最新的為準。\n"
             "輸出兩行：第一行「編號 N」或「不知道」；第二行用一句話回答使用者（不知道就說不記得）。")
if os.environ.get("RR_SYS") == "sound":
    SYS = SYS_SOUND
# H31b: the strict reader stays; only when it says it does not know, rows whose name sounds exactly like a word in the
# question (toneless pinyin, the confusable initials and finals of H25 merged; one syllable may differ for names of
# four or more syllables) are read again with a note that the two spellings are one word.
PHON = os.environ.get("RR_SYS") == "phon"
# H33: the reader also sees the two lines heard just before the question (E43: "他是不是對什麼東西過敏" had the right
# row first and the reader said it did not know; 他 is the person of the previous sentence). With a personal pronoun in
# the question, retrieval also uses the previous owner sentence. RR_ASR = asr.json of the script set.
# RR_CTX=1: context for every question; RR_CTX=pron (H33b): only for questions with a personal pronoun
CTX = os.environ.get("RR_CTX") in ("1", "pron")
CTX_PRON_ONLY = os.environ.get("RR_CTX") == "pron"
PRON = re.compile(r"[他她它牠]")
CTX_RULE = "4. 問話裡的他、她、它或「那個」如果是指前文提到的人或東西，先用前文補上再判斷；前文只用來理解問話，不是記憶紀錄。\n"
if CTX:
    SYS = SYS.replace("輸出兩行：", CTX_RULE + "輸出兩行：")
    assert CTX_RULE in SYS
    _ASR = {d["day"]: d for d in json.load(open(os.environ["RR_ASR"], encoding="utf-8"))}


def before(day, start):
    return [h for h in _ASR[day]["heard"] if h["start"] < start - 0.3][-2:]


def sound_twins(question, rows):
    from label_finder_phon import syllables
    qc = [(ch, (syllables(ch) or [""])[0]) for ch in question if "一" <= ch <= "鿿"]
    out = []
    for i, e in rows:
        ec = [ch for ch in e if "一" <= ch <= "鿿"]
        if len(ec) < 2 or len(ec) != len(e):
            continue
        es = [(syllables(ch) or [""])[0] for ch in ec]
        n = len(es)
        for a in range(len(qc) - n + 1):
            win = qc[a:a + n]
            diff = sum(x[1] != y for x, y in zip(win, es))
            sub = "".join(x[0] for x in win)
            if sub != e and (diff == 0 or (n >= 4 and diff == 1)):
                out.append((i, sub))
                break
    return out


def chat(messages, max_tokens=60):
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, ensure_ascii=False).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception:
            if i == 3:
                raise
            time.sleep(5)


def row_text(ev):
    e, s = ev["label"]
    return f"{e}｜{SLOT_ZH.get(s, s)}｜{ev['value']}（原話：{ev['heard']}）"


def run(path, enc, ks):
    d = json.load(open(path, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(os.path.join(lf.FM, "timeline.json"), encoding="utf-8"))}
    log = [{"label": (w["fact"][0], w["fact"][1]), "value": str(w["fact"][2]), "day": w["day"], "start": w["start"],
            "src": w["src"], "heard": w.get("heard") or ""} for w in d["writes"]]
    R = enc.encode([row_text(ev) for ev in log], normalize_embeddings=True, batch_size=32)
    res = {k: dict(ok=0, wrong=0, none=0, not_routed=0, never_answered=0, never=0, told=0) for k in ks}
    phon_used = {k: 0 for k in ks}
    examples = []
    for q in d["questions"]:
        qq = q["q"]
        told = qq["expect"] is not None
        seg = next((g for g in tl[qq["day"]]["segments"] if g["text"] == qq["text"]), None)
        cand = [] if seg is None else [a for a in d["answers"] if a["day"] == qq["day"] and
                                       min(a["end"], seg["end"]) - max(a["start"], seg["start"]) > 0.3]
        a = max(cand, key=lambda a: lf.overlap(qq["entity"], (a["anchor"] or "").split("|")[0])) if cand else None
        for k in ks:
            res[k]["told" if told else "never"] += 1
        if a is None:
            for k in ks:
                res[k]["not_routed"] += told
            continue
        heard_q = a.get("heard") or qq["text"]
        idx = [i for i, ev in enumerate(log) if ev["day"] < qq["day"] or (ev["day"] == qq["day"] and ev["src"] == "listen"
                                                                          and ev["start"] < seg["start"])]
        if not idx:
            for k in ks:
                res[k]["none"] += told
            continue
        prev = before(qq["day"], seg["start"]) if CTX and (not CTX_PRON_ONLY or PRON.search(heard_q)) else []
        ctx_text = "\n".join(("[使用者] " if h["owner"] == "owner" else "[其他人] ") + h["text"] for h in prev)
        prev_owner = " ".join(h["text"] for h in prev if h["owner"] == "owner")
        qtext = (prev_owner + " " + heard_q) if (CTX and prev_owner and PRON.search(heard_q)) else heard_q
        qv = enc.encode(INSTR + qtext, normalize_embeddings=True)
        sims = R[idx] @ qv
        order = [idx[j] for j in np.argsort(-sims)]
        for k in ks:
            top = sorted(order[:k], key=lambda i: (log[i]["day"], log[i]["start"]))
            lines = [f"{n + 1}. 第{log[i]['day']}天｜{row_text(log[i])}" for n, i in enumerate(top)]
            user = (f"今天是第{qq['day']}天。\n" + (f"前文（剛剛聽到的）：\n{ctx_text}\n" if ctx_text else "")
                    + f"使用者問：{heard_q}\n記憶紀錄：\n" + "\n".join(lines))
            out = chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            m = re.search(r"編號\s*(\d+)", out)
            pick = top[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= len(top) else None
            if (pick is None or UNKNOWN.search(out.split("\n")[0])) and PHON:
                twins = sound_twins(heard_q, [(i, log[i]["label"][0]) for i in top])
                if twins:          # H31b: read again, only the rows whose name sounds the same as a word in the question
                    lines2 = [f"{n + 1}. 第{log[i]['day']}天｜{row_text(log[i])}" for n, (i, _) in enumerate(twins)]
                    note = "；".join(f"問句裡的「{sub}」和紀錄裡的「{log[i]['label'][0]}」唸起來一樣" for i, sub in twins)
                    user2 = (f"今天是第{qq['day']}天。\n使用者問：{heard_q}\n（注意：{note}，是語音辨識聽錯，指的是同一個東西。"
                             f"還是要問的那一方面對得上才回答。）\n記憶紀錄：\n" + "\n".join(lines2))
                    out = chat([{"role": "system", "content": SYS}, {"role": "user", "content": user2}])
                    m = re.search(r"編號\s*(\d+)", out)
                    pick = twins[int(m.group(1)) - 1][0] if m and 1 <= int(m.group(1)) <= len(twins) else None
                    phon_used[k] += 1
            reply = out.split("\n", 1)[1].strip() if "\n" in out else ""
            if pick is None or UNKNOWN.search(out.split("\n")[0]):
                if told:
                    res[k]["none"] += 1
                continue
            if not told:
                res[k]["never_answered"] += 1
                examples.append(("never", k, heard_q, row_text(log[pick])))
                continue
            ans = log[pick]["value"] + " " + reply
            ok = lf.overlap(norm(qq["expect"]), norm(ans)) >= 0.6
            res[k]["ok" if ok else "wrong"] += 1
            if not ok:
                examples.append(("wrong", k, heard_q, row_text(log[pick]), qq["expect"]))
    print("==", os.path.basename(path), flush=True)
    for k in ks:
        r = res[k]
        print(f"  k={k}: right {r['ok']}/{r['told']} | wrong {r['wrong']} | none {r['none']} | not routed "
              f"{r['not_routed']} || never-told answered {r['never_answered']}/{r['never']}"
              + (f" || read again on a sound twin {phon_used[k]}" if PHON else ""), flush=True)
    for x in examples:
        print("   ", x)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    ks = [int(x) for x in os.environ.get("RR_K", "3,5").split(",")]
    for p in sys.argv[1:]:
        run(p, enc, ks)
