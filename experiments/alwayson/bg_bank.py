"""背景聲音庫：把電視、家人閒聊、訪客、家中雜音真的合成成聲音，再用跟主管線相同的感知（faster-whisper
large-v3-turbo + 使用者聲紋）聽一遍。模擬器在沒人跟 Aria 說話的時段，就播放這些「真的聽到的結果」。

為什麼不直接放文字：植物人循環和假喚醒常常是背景雜訊引起的（Gemini 第 10 輪 Q2）。風扇聲會讓 whisper 產生幻聽字幕，
電視聲有時會被聲紋誤認成使用者。這些錯誤只有真的跑一次感知才會出現。

聲音（Windows OneCore zh-TW，與 realistic/render_audio.py 相同）：電視 = Yating（較快、較高、限頻、較小聲），
家人 = Hanhan，訪客 = Yating 正常速度。雜音由程式合成：風扇、冷氣、冰箱、雨聲、街上機車、掃地機器人、洗衣機、提示音。
輸出 bg_bank.json：每個片段 {id, type, dur, heard: [{start, end, text, owner, score}]}。
"""
import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "sensors"))
sys.path.insert(0, HERE)
from bg_texts import FAMILY_TALK, TV_ALL, VISITOR_TALK  # noqa: E402

SR = 16000
CACHE = os.path.join(HERE, "audio", "tts_cache")
PS1 = os.path.join(EXP, "realistic", "tts_onecore.ps1")       # 只讀取使用，不修改
ENROLL = ["我今天先把這幾句話念給你聽，讓你記得我的聲音。", "早安，今天天氣看起來還不錯，等一下要出門買東西。",
          "如果之後有什麼事情我會再跟你說，你先幫我記著就好。"]


def tts(text, voice, rate, pitch):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{text}|{voice}|{rate:.2f}|{pitch:.2f}".encode()).hexdigest()[:16] + ".wav"
    out = os.path.join(CACHE, key)
    if not os.path.exists(out):
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS1, "-Text", text,
                        "-Voice", voice, "-Out", out, "-Rate", str(rate), "-Pitch", str(pitch)],
                       capture_output=True, text=True, check=True)
    w, sr = sf.read(out, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        idx = np.linspace(0, len(w) - 1, int(len(w) * SR / sr))
        w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
    return w


def colored(n, rng, power):
    """power 0 = 白噪音, 1 = 粉紅, 2 = 棕色"""
    x = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(x)
    f = np.arange(len(X), dtype=np.float64)
    f[0] = 1
    return np.fft.irfft(X / f ** (power / 2), n).astype(np.float32)


def at_db(x, db):
    return x * (10 ** (db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9))


def bandlimit(w, lo=300, hi=3400):
    X = np.fft.rfft(w)
    fr = np.fft.rfftfreq(len(w), 1 / SR)
    X[(fr < lo) | (fr > hi)] *= 0.1
    return np.fft.irfft(X, len(w)).astype(np.float32)


def tone(freqs, dur, rng):
    t = np.arange(int(dur * SR)) / SR
    w = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return (w * np.hanning(len(t)) ** 0.2).astype(np.float32)


def room(n, rng):
    return at_db(colored(n, rng, 1), -35)


def noise_clip(kind, rng, dur=30.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = room(n, rng)
    if kind == "fan":
        x += at_db(colored(n, rng, 1.6), rng.uniform(-30, -22)) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t))
    elif kind == "ac":
        hum = np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)
        x += at_db(hum.astype(np.float32), -34) + at_db(colored(n, rng, 1.2), -28)
    elif kind == "fridge":
        hum = np.sin(2 * np.pi * 60 * t) * (t > rng.uniform(2, 10))
        x += at_db(hum.astype(np.float32), -32)
        c = int(rng.uniform(5, 25) * SR)
        x[c:c + 800] += 0.08 * np.hanning(800).astype(np.float32)
    elif kind == "rain":
        x += at_db(bandlimit(colored(n, rng, 0.3), 800, 7000), rng.uniform(-28, -20))
    elif kind == "street":
        for _ in range(rng.integers(1, 4)):        # 機車經過：聲音由小變大再變小
            c, L = rng.uniform(2, dur - 6), rng.uniform(3, 6)
            a, b = int(c * SR), int((c + L) * SR)
            env = np.hanning(b - a).astype(np.float32)
            eng = np.sin(2 * np.pi * rng.uniform(80, 140) * t[a:b]) + 0.3 * colored(b - a, rng, 1)
            x[a:b] += at_db(eng.astype(np.float32), rng.uniform(-24, -16)) * env
    elif kind == "vacuum":
        motor = np.sin(2 * np.pi * rng.uniform(180, 260) * t) * 0.4 + colored(n, rng, 0.8)
        x += at_db(motor.astype(np.float32), rng.uniform(-24, -18)) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * t))
    elif kind == "washer":
        x += at_db((np.sin(2 * np.pi * 50 * t) * (1 + np.sin(2 * np.pi * 0.8 * t))).astype(np.float32), -26)
        x += at_db(colored(n, rng, 1.8), -28)
    elif kind == "beeps":                            # 微波爐、門鈴、手機鈴聲
        for _ in range(rng.integers(1, 3)):
            c = int(rng.uniform(1, dur - 4) * SR)
            pat = rng.choice(["micro", "door", "ring"])
            if pat == "micro":
                w = np.concatenate([np.concatenate([tone([2000], 0.25, rng), np.zeros(int(0.25 * SR), np.float32)]) for _ in range(3)])
            elif pat == "door":
                w = np.concatenate([tone([660, 990], 0.6, rng), tone([520, 780], 0.9, rng)])
            else:
                w = np.concatenate([tone([1300, 1600], 0.4, rng), np.zeros(int(0.2 * SR), np.float32)] * 4)
            x[c:c + len(w)] += 0.15 * w[: n - c]
    elif kind == "quiet":
        pass
    return x


def speech_clip(lines, voice, rate, pitch, gain, rng, tv=False, music=False, under_tv=None):
    parts, t = [], rng.uniform(0.5, 2.0)
    for s in lines:
        w = tts(s, voice, rate * rng.uniform(0.95, 1.05), pitch) * gain * rng.uniform(0.85, 1.1)
        if tv:
            w = bandlimit(w)
        parts.append((t, w))
        t += len(w) / SR + rng.uniform(0.3, 2.5 if not tv else 1.0)
    n = int((t + 1.0) * SR)
    x = room(n, rng)
    for s0, w in parts:
        a = int(s0 * SR)
        x[a:a + len(w)] += w[: n - a]
    if music:                                        # 節目配樂：幾個和弦音，很小聲
        tt = np.arange(n) / SR
        chord = sum(np.sin(2 * np.pi * f * tt) for f in rng.choice([[262, 330, 392], [220, 277, 330], [196, 247, 294]]))
        x += at_db(bandlimit(chord.astype(np.float32)), -32)
    if under_tv is not None:                         # 家人講話時電視還開著
        m = min(n, len(under_tv))
        x[:m] += under_tv[:m]
    return x


def main(seed=0):
    import torch
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    clips = []
    # 電視：每個片段 3-5 句，同一句會出現在不同片段、不同混音裡
    for i in range(72):
        lines = prng.sample(TV_ALL, prng.randint(3, 5))
        clips.append(("tv", speech_clip(lines, "Yating", 1.25, 1.15, prng.uniform(0.3, 0.5), rng, tv=True,
                                        music=prng.random() < 0.4), lines))
    for i in range(30):
        lines = prng.sample(FAMILY_TALK, prng.randint(1, 3))
        under = None
        if prng.random() < 0.35:
            under = speech_clip(prng.sample(TV_ALL, 2), "Yating", 1.25, 1.15, 0.25, rng, tv=True)
        clips.append(("family", speech_clip(lines, "Hanhan", 1.0, 1.0, 0.8, rng, under_tv=under), lines))
    for i in range(10):
        lines = prng.sample(VISITOR_TALK, prng.randint(1, 2))
        clips.append(("visitor", speech_clip(lines, "Yating", 0.95, 0.95, 0.7, rng), lines))
    for kind, nvar in (("fan", 10), ("ac", 6), ("fridge", 6), ("rain", 6), ("street", 10), ("vacuum", 6),
                       ("washer", 4), ("beeps", 8), ("quiet", 10)):
        for i in range(nvar):
            clips.append((kind, noise_clip(kind, rng), []))
    print("rendered", len(clips), "clips", flush=True)

    from faster_whisper import WhisperModel
    from owner_voice import OwnerVoice
    m = WhisperModel(os.environ.get("ASR_MODEL", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"), device="cuda",
                     compute_type="float16")
    ov = OwnerVoice(adapt=0.0)            # 不讓聲紋中心隨片段順序漂移，每個片段的標籤互不影響
    print("enroll", [round(x, 3) for x in ov.enroll([tts(t, "Zhiwei", 1.0, 1.0) for t in ENROLL])], flush=True)
    out = []
    os.makedirs(os.path.join(HERE, "audio"), exist_ok=True)
    for ci, (kind, x, lines) in enumerate(clips):
        x = x / max(1.0, float(np.abs(x).max()) / 0.95)
        segs, _ = m.transcribe(x, language="zh", vad_filter=True, initial_prompt="以下是繁體中文的日常對話，助理的名字叫 Aria。",
                               hotwords="Aria", vad_parameters={"min_silence_duration_ms": 400})
        heard = []
        for s in segs:
            text = s.text.strip()
            if not text:
                continue
            lab, sc = ov.verify(x[int(s.start * SR):int(s.end * SR)])
            heard.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text, "owner": lab,
                          "score": None if sc is None else round(sc, 3)})
        out.append({"id": ci, "type": kind, "dur": round(len(x) / SR, 2), "lines": lines, "heard": heard})
        if ci % 20 == 0:
            print(ci, kind, len(heard), [h["text"][:20] for h in heard[:2]], flush=True)
    json.dump(out, open(os.path.join(HERE, "bg_bank.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # 摘要：各類型產生多少「聽到的句子」、其中多少被聲紋認成使用者
    summ = {}
    for c in out:
        d = summ.setdefault(c["type"], {"clips": 0, "heard": 0, "owner": 0, "unsure": 0})
        d["clips"] += 1
        d["heard"] += len(c["heard"])
        d["owner"] += sum(h["owner"] == "owner" for h in c["heard"])
        d["unsure"] += sum(h["owner"] == "unsure" for h in c["heard"])
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)
    json.dump(summ, open(os.path.join(HERE, "bg_bank_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
