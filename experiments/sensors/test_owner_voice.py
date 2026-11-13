"""Owner verification on synthetic voices (Windows TTS). Weak test: TTS voices are cleaner and more distinct than
people. Enroll with one voice, then score held-out sentences of the same voice and of other voices, including
the same TTS voices with shifted rate/pitch (look-alike speakers)."""
import glob
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from owner_voice import OwnerVoice

HERE = os.path.dirname(os.path.abspath(__file__))
SENTS = ["Hey Aria, I left my keys on the windowsill.", "What's the weather like tomorrow?",
         "Remind me to call my sister tonight.", "I think I put the passport in the drawer.",
         "Can you turn the music down a bit?", "Where did I put my glasses again?"]


def tts(text, voice_idx, rate=0, pitch=0):
    out = tempfile.mktemp(suffix=".wav")
    ssml = (f"<speak version='1.0' xml:lang='en-US'><prosody pitch='{pitch:+d}%' rate='{rate:+d}%'>{text}</prosody></speak>")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$v = $s.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo.Name }}; $s.SelectVoice($v[{voice_idx} % $v.Count]); "
          f"$s.SetOutputToWaveFile('{out}'); $s.SpeakSsml(\"{ssml}\"); $s.Dispose(); $v.Count")
    n = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    w, sr = sf.read(out)
    os.remove(out)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        idx = np.linspace(0, len(w) - 1, int(len(w) * 16000 / sr))
        w = np.interp(idx, np.arange(len(w)), w)
    return w.astype(np.float32), int(n.splitlines()[-1]) if n else 0


def main():
    ov = OwnerVoice()
    _, nvoices = tts("hi", 0)
    print("installed TTS voices:", nvoices)
    enroll = [tts(s, 0)[0] for s in SENTS[:3]]
    print("enroll self-scores", [round(x, 3) for x in ov.enroll(enroll)])
    rows = []
    for s in SENTS[3:]:
        rows.append(("owner held-out", ov.verify(tts(s, 0)[0])))
        rows.append(("owner tired (rate -15%, pitch -5%)", ov.verify(tts(s, 0, rate=-15, pitch=-5)[0])))
        for v in range(1, nvoices):
            rows.append((f"other voice {v}", ov.verify(tts(s, v)[0])))
            rows.append((f"other voice {v} pitch-shifted toward owner", ov.verify(tts(s, v, pitch=-10)[0])))
        rows.append(("owner voice, pitch +20% (look-alike)", ov.verify(tts(s, 0, pitch=20)[0])))
    for name, (lab, sc) in rows:
        print(f"{name:42s} {lab:7s} {sc:.3f}")


if __name__ == "__main__":
    main()
