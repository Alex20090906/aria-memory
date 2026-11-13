"""World v6: single user, Taiwan Mandarin, what model 1 really receives after speech recognition.

Events (one per tick, ~1 s): "[MIC] owner: <text>" or "[MIC] someone: <text>" (owner-voice verification label, with
its measured errors: 3% of owner speech labelled someone, 30% of others labelled owner), "[CAM] <caption>" (English
BLIP-style captions), "[THINKER] ..." when a thought is ready. Text gets ASR-like corruption: no punctuation,
欸->A, 喔->O, 啊->阿, homophone swaps, dropped characters, two utterances merged into one (10%).
Labels follow the single-user rule: only the owner's speech can make model 1 act; the owner's own facts are written
even when said to someone else; nobody else's facts are written.
"""
import json
import os
import random

from world2 import ACTIONS  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REFLEX = os.environ.get("W6_REFLEX", "0") == "1"   # model 1 as reflex layer: owner speech -> ATTEND (listen)
LAB = {"owner_fact": ("REPLY", 1, "owner"), "owner_question": ("THINK", 1, "owner"),
       "owner_request": ("THINK", 1, "owner"), "owner_simple": ("REPLY", 0, "owner"),
       "owner_overheard": ("WAIT", 1, "owner"), "owner_chat_other": ("WAIT", 0, "owner"),
       "other_talk": ("WAIT", 0, "other"), "tv": ("WAIT", 0, "other"), "asr_junk": ("WAIT", 0, "other")}
RATE = {"owner_fact": 0.010, "owner_question": 0.008, "owner_request": 0.005, "owner_simple": 0.006,
        "owner_overheard": 0.008, "owner_chat_other": 0.010, "other_talk": 0.015, "tv": 0.025, "asr_junk": 0.012}
HOMO = [("在", "再"), ("的", "得"), ("那", "哪"), ("他", "她"), ("做", "作"), ("已", "以"), ("帳", "賬"), ("裡", "里"),
        ("訂", "定"), ("幫", "邦"), ("記", "紀"), ("號", "好"), ("點", "典"), ("找", "照"), ("放", "方")]
SCENES = ["a man sitting on a couch watching tv", "an empty kitchen with a fridge", "a cat sleeping on a bed",
          "a person cooking at a stove", "a dog walking through a living room", "a woman reading a book at a table",
          "a package on the floor near a door", "a man carrying a box", "two people talking in a kitchen",
          "a laptop on a desk", "a person putting shoes on", "a child playing with toys", "an open window with curtains",
          "a person holding a phone", "a bicycle leaning on a wall", "a delivery person at the door"]


def asr_noise(s, rng):
    s = s.replace("，", "").replace("。", "").replace("！", "").replace("？", "").replace("、", "")
    s = s.replace("欸", "A" if rng.random() < 0.5 else "欸").replace("喔", "O" if rng.random() < 0.4 else "喔")
    s = s.replace("啊", "阿" if rng.random() < 0.5 else "啊")
    out = []
    for ch in s:
        r = rng.random()
        if r < 0.03:
            continue
        if r < 0.08:
            for a, b in HOMO:
                if ch == a:
                    ch = b
                    break
        out.append(ch)
    return "".join(out)


def split_pools(pools, part, seed=999):
    rng = random.Random(seed)
    out = {}
    for k, xs in pools.items():
        xs = list(xs)
        rng.shuffle(xs)
        cut = int(len(xs) * 0.75)
        out[k] = xs[:cut] if part == "train" else xs[cut:]
    return out


def generate(n_ticks, seed, part):
    pools = split_pools(json.load(open(os.path.join(HERE, "pools_zh.json"), encoding="utf-8")), part)
    rng = random.Random(seed)
    scenes = SCENES[:12] if part == "train" else SCENES
    ticks, busy_for, last_scene = [], 0, {}
    for t in range(n_ticks):
        x = {"t": t, "event": None, "busy": int(busy_for > 0), "busy_for": busy_for, "ideal": "WAIT", "encode": 0,
             "kind": "silence"}
        if busy_for > 0:
            busy_for -= 1
            if busy_for == 0:
                x.update(event="[THINKER] 想好了", ideal="REPLY", kind="result")
            elif rng.random() < 0.02 and pools.get("owner_cancel"):
                x.update(event="[MIC] owner: " + asr_noise(rng.choice(pools["owner_cancel"]), rng), ideal="INTERRUPT",
                         kind="cancel")
                busy_for = 0
            ticks.append(x)
            continue
        r = rng.random()
        acc = 0.0
        for cat, rate in RATE.items():
            acc += rate
            if r < acc and pools.get(cat):
                ideal, enc, who = LAB[cat]
                if REFLEX:
                    ideal, enc = ("ATTEND", 1) if who == "owner" else ("WAIT", 0)
                text = asr_noise(rng.choice(pools[cat]), rng)
                if rng.random() < 0.10:                       # ASR merged two utterances
                    other = rng.choice(list(RATE))
                    if pools.get(other):
                        text = text + " " + asr_noise(rng.choice(pools[other]), rng)
                flip = 0.03 if who == "owner" else 0.30
                label = ("someone" if who == "owner" else "owner") if rng.random() < flip else who.replace("other", "someone")
                x.update(event=f"[MIC] {label}: {text}", ideal=ideal, encode=enc, kind=cat)
                if ideal == "THINK" or (REFLEX and cat in ("owner_question", "owner_request")):
                    busy_for = rng.randint(4, 20)
                break
        else:
            if r < acc + 0.01:
                sc = rng.choice(scenes)
                last = last_scene.get(sc)
                if last is None:
                    ideal, enc, kind = "ATTEND", 1, "scene"
                elif t - last <= 250:
                    ideal, enc, kind = "WAIT", 0, "scene_repeat"
                elif t - last >= 1500:
                    ideal, enc, kind = "ATTEND", 1, "scene_return"
                else:
                    sc = None
                if sc:
                    last_scene[sc] = t
                    x.update(event="[CAM] " + sc, ideal=ideal, encode=enc, kind=kind)
        ticks.append(x)
    return ticks


if __name__ == "__main__":
    from collections import Counter
    t = generate(30000, 0, "test")
    print(Counter(x["kind"] for x in t))
    for x in t[:400]:
        if x["event"]:
            print(x["ideal"], x["encode"], x["event"][:70])
