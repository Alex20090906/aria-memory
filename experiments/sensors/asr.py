"""Hearing for model 1: speech -> timestamped text events "[MIC] ...".

Two sources with the same output:
  file  a wav file, processed in 1 s ticks as if it were live (for tests, no microphone needed)
  live  a microphone through ffmpeg (dshow); only started on request

Speech is cut into utterances by faster-whisper's VAD; each finished utterance becomes one event at the tick
where it ended.  Who spoke is unknown here (no speaker recognition yet), so events read "[MIC] someone: ...".
"""
import argparse
import json
import subprocess
import sys

import numpy as np

SR = 16000


def load_wav(path):
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768


class Speakers:
    """Who is talking, without names: a voice embedding per utterance (resemblyzer), matched to known voices by
    cosine (>= thresh -> same speaker, its centroid updated; else a new speaker#N).  Names are not known here: the
    binding speaker#N -> "Leo" is a memory fact learned later (someone says "I'm Leo", or is addressed by name)."""

    def __init__(self, thresh=0.75):
        from resemblyzer import VoiceEncoder
        self.enc = VoiceEncoder("cpu", verbose=False)
        self.cent, self.n, self.thresh = [], [], thresh

    def identify(self, wav16k):
        if len(wav16k) < SR // 2:
            return "someone"
        e = self.enc.embed_utterance(wav16k.astype(np.float32))
        if self.cent:
            sims = np.array([float(e @ c) / (np.linalg.norm(c) + 1e-6) for c in self.cent])
            i = int(sims.argmax())
            if sims[i] >= self.thresh:
                self.cent[i] = (self.cent[i] * self.n[i] + e) / (self.n[i] + 1)
                self.n[i] += 1
                return f"speaker#{i + 1}"
        self.cent.append(e.copy())
        self.n.append(1)
        return f"speaker#{len(self.cent)}"


class Hearing:
    def __init__(self, model="small", device="cuda", language=None, speakers=True):
        import os
        import torch  # its wheel carries the CUDA 12 cuBLAS/cuDNN DLLs that CTranslate2 needs
        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
        os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")
        self.language = language
        self.spk = Speakers() if speakers else None

    def utterances(self, audio, t0=0.0):
        segs, _ = self.m.transcribe(audio, vad_filter=True, language=self.language,
                                    vad_parameters={"min_silence_duration_ms": 500})
        for s in segs:
            text = s.text.strip()
            if text:
                who = self.spk.identify(audio[int(s.start * SR):int(s.end * SR)]) if self.spk else "someone"
                yield {"t": int(t0 + s.end + 0.999), "start": round(t0 + s.start, 2), "speaker": who,
                       "event": f"[MIC] {who}: {text}"}


def run_file(path, model):
    h = Hearing(model)
    audio = load_wav(path)
    return list(h.utterances(audio))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--model", default="small")
    a = p.parse_args()
    for e in run_file(a.wav, a.model):
        print(json.dumps(e, ensure_ascii=False))
