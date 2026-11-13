"""Why some owner questions never reach an answer in zh_v3 (E42 "not routed"): replay the listening call for the
owner-heard segments of those questions against the normal LLM server service and print the raw output.
Context is reconstructed from asr.json (the 6 previous heard lines); TOPIC and the staging list are not known
offline and are given as empty, so this is a diagnosis, not a score.

    python route_diag.py <zh_result.json> <timeline.json> <asr.json>
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "..", "llm_fastmem", "zh_v3.py"), encoding="utf-8").read()
NS = {"SLOTS": ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"],
      "CANON": False, "os": os, "re": re}
block = SRC[SRC.index("STAGE_SYS = ("):SRC.index("STAGE_RE = ")] + SRC[SRC.index("LISTEN_SYS = ("):SRC.index("RELISTEN_SYS")]
exec(block, NS)
WD = "一二三四五六日"


def main(res, tlp, asrp):
    d = json.load(open(res, encoding="utf-8"))
    tl = {x["day"]: x for x in json.load(open(tlp, encoding="utf-8"))}
    asr = {x["day"]: x for x in json.load(open(asrp, encoding="utf-8"))}
    for r in d["questions"]:
        if r["routed"]:
            continue
        q = r["q"]
        seg = next((g for g in tl[q["day"]]["segments"] if g["text"] == q["text"]), None)
        heard = asr[q["day"]]["heard"]
        for i, h in enumerate(heard):
            if h["owner"] != "owner" or not seg or min(h["end"], seg["end"]) - max(h["start"], seg["start"]) <= 0.3:
                continue
            ctx = [("[使用者] " if x["owner"] == "owner" else "[其他人] ") + x["text"] for x in heard[max(0, i - 5):i + 1]]
            dd = f"今天是第{q['day']}天（星期{WD[(q['day'] - 1) % 7]}）。"
            text = (dd + "\nTOPIC：（無）\n前文：\n" + ("\n".join(ctx[:-1]) or "（無）") + "\n暫存：\n（無）"
                    + "\n剛聽到的一句：\n" + ctx[-1])
            out = chat_shots(NS["LISTEN_SYS"] + NS["STAGE_SYS"], NS["LISTEN_SHOTS"], text)
            print(f"day {q['day']} {'TOLD' if q['expect'] else 'never'} | heard: {h['text']}\n   -> {out}", flush=True)


def chat_shots(system, shots, text):
    import time
    import urllib.request
    msgs = [{"role": "system", "content": system}]
    for a, b in shots:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": text})
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": msgs, "max_tokens": 160, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}, ensure_ascii=False).encode()
    req = urllib.request.Request("http://<thinker-host>:8081/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return (json.loads(r.read())["choices"][0]["message"]["content"] or "").strip().splitlines()
        except Exception:
            if i == 3:
                raise
            time.sleep(5)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:4])
