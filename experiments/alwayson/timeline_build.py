"""把 21 天的生活腳本放進真實時鐘，做成常駐模擬器要播放的多天時間線。

來源（全部是之前真的跑過感知的結果，沒有改寫）：
  llm_fastmem/zh_scripts.json   Gemini 寫的台灣口語雜亂腳本（21 天、200 段、90 個事實、72 題）
  llm_fastmem/timeline.json     每一段在當天錄音裡的真實起訖時間
  llm_fastmem/asr.json          large-v3-turbo 語音辨識＋使用者聲紋的結果（系統真正「聽到」的）
  alwayson/bg_bank.json            電視、家人、訪客、家中雜音，同一套感知聽過的結果

做法：每天的腳本段落照原順序分成三段（早上、傍晚、晚上），段內保留原本的相對時間；其餘時間依作息放背景：
平日 08:30–18:40 使用者上班不在家，晚上看電視；週末白天在家、下午出門。攝影機事件用 BLIP 風格的英文描述（只用合成畫面）。
每一個事件都帶「真實來源」（誰說的、對誰說），只用來評分，系統看不到。
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SF = os.path.join(os.path.dirname(HERE), "llm_fastmem")


def hm(h, m=0):
    return h * 3600 + m * 60


CAM = {
    "wake": "a person turning on the lights in a bedroom", "kitchen": "a person cooking at a stove",
    "breakfast": "a man eating breakfast at a table", "shoes": "a person putting shoes on",
    "empty": "an empty living room with a sofa", "vacuum": "a robot vacuum cleaner on the floor",
    "delivery": "a delivery person at the door", "family_in": "a woman walking into the living room",
    "owner_in": "a man walking into the living room carrying a bag", "tv": "a man sitting on a couch watching tv",
    "dinner": "two people eating dinner at a table", "bath": "a person brushing teeth in a bathroom",
    "dark": "a dark room at night", "curtain": "an open window with curtains", "cat": "a cat sleeping on a sofa",
    "laptop": "a man using a laptop on a desk", "visitor": "two people talking in a living room",
}


def main(days=21, seed=0, out_name="timeline_21d.json"):
    rng = random.Random(seed)
    sc = json.load(open(os.path.join(SF, "zh_scripts.json"), encoding="utf-8"))
    tl = {d["day"]: d for d in json.load(open(os.path.join(SF, "timeline.json"), encoding="utf-8"))}
    asr = {d["day"]: d for d in json.load(open(os.path.join(SF, "asr.json"), encoding="utf-8"))}
    bank = json.load(open(os.path.join(HERE, "bg_bank.json"), encoding="utf-8"))
    by_type = {}
    for c in bank:
        by_type.setdefault(c["type"], []).append(c)
    qtexts = {(q["day"], q["text"]) for q in sc["questions"]}
    events, presence, episodes = [], [], []

    for day in sc["days"][:days]:
        d = day["day"]
        base = (d - 1) * 86400
        weekend = day["weekday"] in ("六", "日")
        occupied = []                                        # 已經有聲音的時段，背景不疊上去

        def cam(t, key):
            events.append({"t": base + t, "kind": "cam", "text": CAM[key], "owner": None, "src": "cam", "day": d,
                           "truth": {"type": "cam", "scene": key}})

        def put_clip(t, clip, kind_override=None):
            if any(a - 5 < t + clip["dur"] and t < b + 5 for a, b in occupied):
                return False
            occupied.append((t, t + clip["dur"]))
            for h in clip["heard"]:
                events.append({"t": base + t + h["end"], "kind": "mic", "text": h["text"], "owner": h["owner"],
                               "score": h["score"], "src": "bg", "day": d,
                               "truth": {"type": kind_override or clip["type"], "clip": clip["id"]}})
            return True

        # 作息
        if weekend:
            wake, sleep_t = hm(8, rng.randint(10, 50)), hm(23, rng.randint(30, 55))
            ep_t = [hm(9, 40), hm(13, 10), hm(21, 0)]
            away = [(hm(14, 30), hm(17, 40))]
            tv_blocks = [(hm(10, 0), hm(10, 40)), (hm(19, 30), hm(20, 40))]
        else:
            wake, sleep_t = hm(6, rng.randint(40, 59)), hm(23, rng.randint(10, 40))
            ep_t = [hm(7, 15), hm(19, 5), hm(21, 40)]
            away = [(hm(8, 30), hm(18, 40))]
            tv_blocks = [(hm(7, 40), hm(7, 55)), (hm(19, 40), hm(20, 40))]
        presence.append({"day": d, "wake": base + wake, "sleep": base + sleep_t,
                         "away": [(base + a, base + b) for a, b in away], "weekend": weekend})

        # 腳本：三段，段內保留真實相對時間
        segs = tl[d]["segments"]
        n = len(segs)
        cut = [0, (n + 2) // 3, (2 * n + 2) // 3, n]
        heard = asr[d]["heard"]
        for e in range(3):
            part = segs[cut[e]:cut[e + 1]]
            if not part:
                continue
            s0, s1 = part[0]["start"] - 1.0, part[-1]["end"] + 1.0
            t0 = ep_t[e]
            occupied.append((t0, t0 + (s1 - s0)))
            episodes.append({"day": d, "t0": base + t0, "t1": base + t0 + (s1 - s0), "segments": [g["i"] for g in part]})
            for h in heard:
                if not (s0 <= h["start"] < s1):
                    continue
                ov = [g for g in segs if min(g["end"], h["end"]) - max(g["start"], h["start"]) > 0.3]
                main_g = max(ov, key=lambda g: min(g["end"], h["end"]) - max(g["start"], h["start"])) if ov else None
                truth = {"type": "script", "speaker": main_g["speaker"] if main_g else "unknown",
                         "to": main_g["to"] if main_g else "none", "seg_i": main_g["i"] if main_g else None,
                         "question": bool(main_g and (d, main_g["text"]) in qtexts)}
                events.append({"t": base + t0 + (h["end"] - s0), "kind": "mic", "text": h["text"], "owner": h["owner"],
                               "score": h["score"], "src": "script", "day": d, "hstart": h["start"], "hend": h["end"],
                               "truth": truth})

        # 攝影機
        cam(wake, "wake")
        cam(wake + 900, "kitchen")
        cam(wake + 1500, "breakfast")
        for a, b in away:
            cam(a - 300, "shoes")
            cam(a + 120, "empty")
            cam(b, "owner_in")
        if not weekend:
            if rng.random() < 0.8:
                cam(hm(10, 30), "vacuum")
                for _ in range(2):
                    put_clip(hm(10, 30) + rng.randint(0, 1200), rng.choice(by_type["vacuum"]))
            cam(hm(18, 0), "family_in")
        cam(hm(19, 30), "tv")
        cam(hm(18, 50) if not weekend else hm(18, 30), "dinner")
        cam(sleep_t - 1200, "bath")
        cam(sleep_t, "dark")
        cam(hm(rng.randint(9, 16), rng.randint(0, 59)), "curtain")
        cam(hm(rng.randint(11, 17), rng.randint(0, 59)), "cat")
        if rng.random() < 0.5:
            t = hm(rng.randint(10, 17), rng.randint(0, 59))
            cam(t, "delivery")
            put_clip(t + 5, rng.choice(by_type["beeps"]))
        if rng.random() < 0.3:
            t = hm(15, rng.randint(0, 59)) if weekend else hm(20, rng.randint(45, 59))
            cam(t, "visitor")
            put_clip(t + 30, rng.choice(by_type["visitor"]))

        # 電視：開著的時段一段接一段播（同一套感知聽到的句子）
        for a, b in tv_blocks:
            t = a
            while t < b:
                clip = rng.choice(by_type["tv"])
                if not put_clip(t, clip):
                    t += 60
                    continue
                t += clip["dur"] + rng.uniform(1, 8)
        # 家人：晚上在家時偶爾自己講話、講電話
        fam_window = (hm(10, 0), hm(22, 30)) if weekend else (hm(18, 0), hm(22, 30))
        for _ in range(rng.randint(3, 6)):
            put_clip(rng.randint(*fam_window), rng.choice(by_type["family"]))
        # 家中雜音：白天與夜裡（風扇、冰箱、冷氣、雨、機車、洗衣機）
        for kind, k_n in (("fan", 6), ("fridge", 6), ("ac", 3), ("street", 6), ("rain", 2), ("washer", 1)):
            for _ in range(k_n):
                put_clip(rng.randint(0, 86000), rng.choice(by_type[kind]))
    events.sort(key=lambda e: e["t"])
    for i, e in enumerate(events):
        e["i"] = i
    out = {"days": days, "events": events, "presence": presence, "episodes": episodes}
    json.dump(out, open(os.path.join(HERE, out_name), "w", encoding="utf-8"), ensure_ascii=False)
    kinds = {}
    for e in events:
        k = e["src"] + ":" + (e["truth"].get("speaker") or e["truth"].get("type"))
        kinds[k] = kinds.get(k, 0) + 1
    print("events", len(events), json.dumps(kinds, ensure_ascii=False))
    print("owner-labelled", sum(1 for e in events if e["owner"] == "owner"),
          "of which not owner speech", sum(1 for e in events if e["owner"] == "owner" and
                                           e["truth"].get("speaker") not in ("owner", "phone_owner")))


if __name__ == "__main__":
    main()
