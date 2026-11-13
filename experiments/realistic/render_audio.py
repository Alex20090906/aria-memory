"""Render zh_scripts.json to one noisy room recording per day (test level 2: synthetic speech).

Voices (Windows OneCore zh-TW): owner = Zhiwei (rate/pitch jitter per utterance), family = Hanhan, visitor = Yating,
TV = Yating faster and higher, band-limited, quieter, sometimes still playing under the next owner utterance.
Room: pink-ish noise at about -35 dB, random gaps 0.4-4 s, 25% of utterances get a fan/clatter burst.
Writes audio/day_XX.wav (16 kHz mono) and audio/timeline.json with the true start/end of every segment.
"""
import hashlib
import json
import os
import random
import subprocess

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, os.environ.get("SET_DIR", ""))   # script set folder (default: this folder)
SR = 16000
VOICE = {"owner": ("Zhiwei", 1.0, 1.0), "phone_owner": ("Zhiwei", 1.0, 1.0), "family": ("Hanhan", 1.0, 1.0),
         "visitor": ("Yating", 0.95, 0.95), "tv": ("Yating", 1.25, 1.15)}
CACHE = os.path.join(HERE, "audio", "tts_cache")


def tts(text, voice, rate, pitch):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{text}|{voice}|{rate:.2f}|{pitch:.2f}".encode()).hexdigest()[:16] + ".wav"
    out = os.path.join(CACHE, key)
    if not os.path.exists(out):
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        os.path.join(HERE, "tts_onecore.ps1"), "-Text", text, "-Voice", voice, "-Out", out,
                        "-Rate", str(rate), "-Pitch", str(pitch)], capture_output=True, text=True, check=True)
    w, sr = sf.read(out, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        idx = np.linspace(0, len(w) - 1, int(len(w) * SR / sr))
        w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
    return w


def pink(n, rng):
    x = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(x)
    f = np.arange(len(X)); f[0] = 1
    return np.fft.irfft(X / np.sqrt(f), n).astype(np.float32)


def bandlimit(w):
    X = np.fft.rfft(w)
    fr = np.fft.rfftfreq(len(w), 1 / SR)
    X[(fr < 300) | (fr > 3400)] *= 0.1
    return np.fft.irfft(X, len(w)).astype(np.float32)


def main(seed=0):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    d = json.load(open(os.path.join(SD, "zh_scripts.json"), encoding="utf-8"))
    os.makedirs(os.path.join(SD, "audio"), exist_ok=True)
    timeline = []
    for day in d["days"]:
        parts, t, segs, tv_tail = [], 0.0, [], None
        for i, s in enumerate(day["segments"]):
            v, r, p = VOICE.get(s["speaker"], VOICE["family"])
            if s["speaker"] in ("owner", "phone_owner"):
                r *= rng.uniform(0.9, 1.15); p *= rng.uniform(0.95, 1.05)
            w = tts(s["text"], v, r, p)
            gain = {"tv": 0.35, "family": 0.8, "visitor": 0.7}.get(s["speaker"], 1.0) * rng.uniform(0.8, 1.1)
            if s["speaker"] == "tv":
                w = bandlimit(w)
            w = w * gain
            gap = rng.uniform(0.4, 4.0)
            start = t + gap
            # TV keeps talking under the next utterance 40% of the time (overlap)
            if tv_tail is not None and rng.random() < 0.4:
                start = max(t - len(tv_tail) / SR * 0.6, t - 2.5)
            parts.append((start, w))
            segs.append({"i": i, "speaker": s["speaker"], "to": s["to"], "text": s["text"],
                         "start": round(start, 2), "end": round(start + len(w) / SR, 2)})
            t = start + len(w) / SR
            tv_tail = w if s["speaker"] == "tv" else None
        total = int((t + 2.0) * SR)
        mix = np.zeros(total, np.float32)
        for start, w in parts:
            a = int(start * SR)
            mix[a:a + len(w)] += w[:max(0, total - a)]
        noise = pink(total, nrng)
        noise *= 10 ** (-35 / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
        mix += noise
        for _ in range(int(len(parts) * 0.25)):          # clatter bursts
            a = nrng.integers(0, max(1, total - SR))
            burst = nrng.standard_normal(int(0.25 * SR)).astype(np.float32) * 0.05 * np.hanning(int(0.25 * SR))
            mix[a:a + len(burst)] += burst
        mix /= max(1.0, np.abs(mix).max() / 0.95)
        path = os.path.join(SD, "audio", f"day_{day['day']:02d}.wav")
        sf.write(path, mix, SR)
        timeline.append({"day": day["day"], "wav": os.path.basename(path), "segments": segs})
        print("day", day["day"], "segments", len(segs), "seconds", round(total / SR), flush=True)
    json.dump(timeline, open(os.path.join(SD, "audio", "timeline.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
