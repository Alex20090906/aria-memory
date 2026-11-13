"""Perception on the rendered days: faster-whisper (zh) + owner-voice verification. Nothing from the script is used
except for scoring afterwards.

Writes asr.json: per day the list of heard utterances {start, end, text, owner: owner|other|unsure, score}.
Scores (against audio/timeline.json): character error rate of owner speech, owner-verification confusion,
segments never heard, TV/others merged into owner utterances.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, os.environ.get("SET_DIR", ""))   # script set folder (default: this folder)
sys.path.insert(0, os.path.join(HERE, "..", "sensors"))
SR = 16000
MODEL = os.environ.get("ASR_MODEL", "small")
SPLIT = os.environ.get("ASR_SPLIT", "0") == "1"           # split segments by per-word voice score
SPLIT_TH = float(os.environ.get("ASR_SPLIT_TH", "0.70"))  # owner/other boundary used only for cutting
ENROLL = ["我今天先把這幾句話念給你聽，讓你記得我的聲音。", "早安，今天天氣看起來還不錯，等一下要出門買東西。",
          "如果之後有什麼事情我會再跟你說，你先幫我記著就好。"]


def cer(ref, hyp):
    r, h = list(ref), list(hyp)
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[len(h)] / max(1, len(r))


def norm(s):
    return "".join(ch for ch in s if ch.isalnum())


def main():
    import torch
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
    from faster_whisper import WhisperModel
    from owner_voice import OwnerVoice
    from render_audio import tts
    m = WhisperModel(MODEL, device="cuda", compute_type="float16")
    ov = OwnerVoice()
    print("enroll", [round(x, 3) for x in ov.enroll([tts(t, "Zhiwei", 1.0, 1.0) for t in ENROLL])], flush=True)
    tl = json.load(open(os.path.join(SD, "audio", "timeline.json"), encoding="utf-8"))
    out, stats = [], {"cer_owner": [], "verify": {}, "missed_owner": 0, "owner_segments": 0}
    for day in tl:
        audio, _ = sf.read(os.path.join(SD, "audio", day["wav"]), dtype="float32")
        segs, _ = m.transcribe(audio, language="zh", vad_filter=True, initial_prompt="以下是繁體中文的日常對話，助理的名字叫 Aria。", hotwords="Aria",
                               vad_parameters={"min_silence_duration_ms": 400}, word_timestamps=SPLIT)
        heard = []
        for s in segs:
            text = s.text.strip()
            if not text:
                continue
            if SPLIT and s.words:
                # Gemini round 8: ASR segments often hold two speakers. Score each word with a 1.5 s voice window,
                # smooth, and cut the segment where the voice changes or where words are > 0.35 s apart.
                ws = [w for w in s.words if w.word.strip()]
                sc_w = []
                for w in ws:
                    c = (w.start + w.end) / 2
                    a0, a1 = max(0, int((c - 0.75) * SR)), min(len(audio), int((c + 0.75) * SR))
                    sc_w.append(ov.score(audio[a0:a1]) or 0.0)
                sm = [float(np.median(sc_w[max(0, i - 1):i + 2])) for i in range(len(sc_w))]
                runs, cur = [], None
                for w, scv in zip(ws, sm):
                    lab_w = scv >= SPLIT_TH
                    if cur and (lab_w == cur["lab"]) and w.start - cur["end"] <= 0.35:
                        cur["words"].append(w.word); cur["end"] = w.end; cur["scores"].append(scv)
                    else:
                        cur = {"lab": lab_w, "start": w.start, "end": w.end, "words": [w.word], "scores": [scv]}
                        runs.append(cur)
                for r in runs:
                    t = "".join(r["words"]).strip()
                    if not t:
                        continue
                    lab, sc = ov.verify(audio[int(r["start"] * SR):int(r["end"] * SR)])
                    heard.append({"start": round(r["start"], 2), "end": round(r["end"], 2), "text": t, "owner": lab,
                                  "score": None if sc is None else round(sc, 3)})
                continue
            lab, sc = ov.verify(audio[int(s.start * SR):int(s.end * SR)])
            heard.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text, "owner": lab,
                          "score": None if sc is None else round(sc, 3)})
        out.append({"day": day["day"], "heard": heard})
        # scoring: match each true segment to heard utterances overlapping it
        for g in day["segments"]:
            ov_h = [h for h in heard if min(h["end"], g["end"]) - max(h["start"], g["start"]) > 0.3]
            truth_owner = g["speaker"] in ("owner", "phone_owner")
            for h in ov_h:
                key = ("owner" if truth_owner else g["speaker"]) + "->" + h["owner"]
                stats["verify"][key] = stats["verify"].get(key, 0) + 1
            if truth_owner:
                stats["owner_segments"] += 1
                if not ov_h:
                    stats["missed_owner"] += 1
                else:
                    stats["cer_owner"].append(cer(norm(g["text"]), norm("".join(h["text"] for h in ov_h))))
        print("day", day["day"], "heard", len(heard), flush=True)
    json.dump(out, open(os.path.join(SD, "asr.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    c = stats.pop("cer_owner")
    stats["owner_cer_mean"] = round(float(np.mean(c)), 3) if c else None
    stats["owner_cer_median"] = round(float(np.median(c)), 3) if c else None
    print("SUMMARY", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
