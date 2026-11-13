"""用 語言模型伺服器 當裁判，重判「早上主動提醒」對不對。

為什麼：analyze.py 的逐字比對太嚴。「第3天下午2:30」和「禮拜三下午兩點半」是同一件事（第 3 天就是禮拜三），
但字元重疊只有 2/8，被算成錯。裁判看的是使用者到那天為止真的講過的事（腳本裡的事實，含改口與更新），
判斷這句提醒是：正確（和使用者最新講的一致）、過時（舊版本，或那件事已經過去）、錯誤（內容不對）、無依據（使用者沒講過）。
只判提醒；問題與跨日探測仍用原本的逐字比對（與之前的研究一致，會低估）。
輸出 results/judge_reminders.json：{run: [{day, e, s, val, text, verdict, reason}]}
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(os.path.dirname(HERE), "llm_fastmem")
sys.path.insert(0, HERE)
from sim import WD, overlap  # noqa: E402

SC = json.load(open(os.path.join(SF, "zh_scripts.json"), encoding="utf-8"))
SYS = ("你是評分員。使用者有一個居家助理。助理在某天早上主動對使用者說了一句提醒。你會拿到使用者到那天為止真的講過的相關事情"
       "（第幾天講的、講了什麼；同一件事後來講的會蓋過前面的）。判斷這句提醒屬於哪一種，只輸出一行「判定 | 理由」，判定只能是："
       "正確（和使用者最新講的一致，時間換算對，例如第3天就是星期三）、過時（用的是舊版本，或那件事在提醒那天之前就已經過去）、"
       "錯誤（內容和使用者講的不符）、無依據（使用者沒講過這件事）。")


def chat(msgs, n=60):
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": msgs, "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "llm_checkpoint": False}
    req = urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            print("retry", e, flush=True)
            time.sleep(5)
    return ""


def related(x):
    day = x["day"]
    facts = [f for f in SC["facts"] if f["day"] <= day]
    rel = [f for f in facts if max(overlap(f["entity"], x["e"]), overlap(x["e"], f["entity"])) >= 0.5
           or (x.get("val") and overlap(f["value"], x["val"]) >= 0.5)]
    if not rel:
        rel = [f for f in facts if f["slot"] in ("time", "plan")]
    return rel


def owner_lines(x):
    """使用者真的講過的原話（腳本，不是聽寫）：腳本只標註了 90 件事，使用者講過但沒標註的事也算有依據"""
    out = []
    for d in SC["days"]:
        if d["day"] > x["day"]:
            break
        for s in d["segments"]:
            if s["speaker"] in ("owner", "phone_owner") and (overlap(x["e"], s["text"]) >= 0.5
                                                             or (x.get("val") and overlap(x["val"], s["text"]) >= 0.5)):
                out.append(f"第{d['day']}天（星期{WD[(d['day'] - 1) % 7]}）使用者說：「{s['text']}」")
    return out[-8:]


def main():
    out_path = os.path.join(HERE, "results", "judge_reminders.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", "run_*.json"))):
        run = os.path.basename(p)[4:-5]
        if run in out:
            continue
        r = json.load(open(p, encoding="utf-8"))
        rows = []
        for x in r["log"]["proactive"]:
            if x["kind"] != "提醒":
                continue
            facts = "\n".join(f"第{f['day']}天（星期{WD[(f['day'] - 1) % 7]}）：{f['entity']} 的 {f['slot']} 是「{f['value']}」"
                              for f in related(x))
            lines = "\n".join(owner_lines(x))
            user = (f"提醒那天：第{x['day']}天（星期{WD[(x['day'] - 1) % 7]}）早上。\n助理說：「{x['text']}」"
                    f"（它記得的是：{x['e']} 的 {x['s']} 是「{x.get('val')}」）\n使用者講過的相關事情（整理過的）：\n{facts or '（沒有）'}"
                    f"\n使用者的原話：\n{lines or '（沒有）'}")
            ans = chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}]).strip().splitlines()
            line = ans[0] if ans else ""
            m = re.match(r"\s*(正確|過時|錯誤|無依據)", line)
            rows.append({"day": x["day"], "e": x["e"], "s": x["s"], "val": x.get("val"), "text": x["text"],
                         "verdict": m.group(1) if m else "?", "reason": line[:120]})
        out[run] = rows
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        c = {}
        for x in rows:
            c[x["verdict"]] = c.get(x["verdict"], 0) + 1
        print(run, c, flush=True)


if __name__ == "__main__":
    main()
