"""H43: echo of the system's own voice (Gemini round 12 checklist: on a real device the microphone hears the speaker).

Aria's spoken replies of the always-on run E49 (answers right after the owner's questions, plus the morning
reminders) are rendered with a Windows zh-TW voice, band-limited like a small speaker, and mixed into the day's room
recording right where the system would have spoken (0.4 s after the heard question ends; reminders 1 s after the
owner's first utterance of the day). The pipeline's own recognizer (faster-whisper large-v3-turbo, CPU int8) and owner
voice check (resemblyzer, enrolled on the owner's voice as in the pipeline) then hear a window around each reply.

Counted: Aria's speech segments heard as the owner (the listener would act on them and could write the system's own
words back into memory), as unsure, as someone else; and how many remain after a half-duplex gate (drop every heard
segment that overlaps the time the system is speaking, plus 0.3 s).

    python echo_sim.py <run json> <audio dir> <asr.json> <voice: Yating|Hanhan|Zhiwei> <out.json>
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "realistic"))
sys.path.insert(0, os.path.join(HERE, "..", "sensors"))
SR = 16000
ENROLL = ["我今天先把這幾句話念給你聽，讓你記得我的聲音。", "早安，今天天氣看起來還不錯，等一下要出門買東西。",
          "如果之後有什麼事情我會再跟你說，你先幫我記著就好。"]
PROMPT = "以下是繁體中文的日常對話，助理的名字叫 Aria。"


def main(run_path, audio_dir, asr_path, voice, out_path):
    from faster_whisper import WhisperModel
    from owner_voice import OwnerVoice
    from render_audio import tts, bandlimit
    run = json.load(open(run_path, encoding="utf-8"))
    tl = {d["day"]: d for d in json.load(open(os.path.join(audio_dir, "timeline.json"), encoding="utf-8"))}
    asr = {d["day"]: d for d in json.load(open(asr_path, encoding="utf-8"))}
    m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8")
    ov = OwnerVoice()
    ov.enroll([tts(t, "Zhiwei", 1.0, 1.0) for t in ENROLL])
    # what the system says and when (day-relative seconds of the day recording)
    says = {}
    for a in run["log"]["answers"]:
        if a.get("hend") is None:
            continue
        text = (a.get("reply") or "").strip() or ("我不記得了。" if not a["passed"] else str(a["tail"]).split(" ", 1)[-1])
        says.setdefault(a["day"], []).append((a["hend"] + 0.4, text[:60]))
    for p in run["log"].get("proactive", []):
        hs = [h for h in asr.get(p["day"], {}).get("heard", []) if h["owner"] == "owner"]
        if hs:
            says.setdefault(p["day"], []).append((hs[0]["end"] + 1.0, str(p.get("text", ""))[:60]))
    res = {"replies": 0, "heard_segments": 0, "owner": 0, "unsure": 0, "other": 0, "owner_after_gate": 0,
           "examples": []}
    t0 = time.time()
    for day in sorted(says):
        audio, _ = sf.read(os.path.join(audio_dir, tl[day]["wav"]), dtype="float32")
        audio = audio.copy()
        spans = []
        for start, text in says[day]:
            w = bandlimit(tts(text, voice, 1.0, 1.0)) * 0.6
            a0 = int(start * SR)
            if a0 >= len(audio):
                continue
            a1 = min(len(audio), a0 + len(w))
            audio[a0:a1] += w[:a1 - a0]
            spans.append((start, a1 / SR, text))
        # the gate's cost: owner speech (as heard in the original run) that overlaps the system's speaking time
        for h in asr[day]["heard"]:
            if h["owner"] == "owner" and any(min(h["end"], s1 + 0.3) - max(h["start"], s0 - 0.3) > 0 for s0, s1, _ in spans):
                res["owner_lost_to_gate"] = res.get("owner_lost_to_gate", 0) + 1
        res["owner_segments_total"] = res.get("owner_segments_total", 0) + sum(h["owner"] == "owner" for h in asr[day]["heard"])
        for s0, s1, text in spans:
            res["replies"] += 1
            w0, w1 = max(0.0, s0 - 1.0), min(len(audio) / SR, s1 + 1.0)
            segs, _ = m.transcribe(audio[int(w0 * SR):int(w1 * SR)], language="zh", vad_filter=True,
                                   initial_prompt=PROMPT, hotwords="Aria", vad_parameters={"min_silence_duration_ms": 400})
            for sg in segs:
                g0, g1 = w0 + sg.start, w0 + sg.end
                if min(g1, s1) - max(g0, s0) <= 0.3:        # this heard segment is not the system's own speech
                    continue
                res["heard_segments"] += 1
                lab, sc = ov.verify(audio[int(g0 * SR):int(g1 * SR)])
                res[lab] += 1
                # half-duplex gate: anything overlapping the system's speaking time (+0.3 s) is dropped
                gated = min(g1, s1 + 0.3) - max(g0, s0 - 0.3) > 0
                if lab == "owner" and not gated:
                    res["owner_after_gate"] += 1
                if lab == "owner" and len(res["examples"]) < 8:
                    res["examples"].append((day, text[:30], sg.text.strip()[:30], None if sc is None else round(sc, 3)))
        print(f"day {day}: replies {res['replies']} heard {res['heard_segments']} as owner {res['owner']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SUMMARY", json.dumps({k: v for k, v in res.items() if k != "examples"}, ensure_ascii=False))
    for e in res["examples"]:
        print("  heard as owner:", e)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:6])
