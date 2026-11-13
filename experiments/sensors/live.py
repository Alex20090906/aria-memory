"""The always-on loop with real senses: microphone + camera -> text events, one tick per second -> model 1.

Nothing here opens a device unless you pass --mic and/or --cam (ffmpeg dshow names, e.g. --mic "麥克風 (2- USB Audio
Device)" --cam "USB Camera").  --wav / --frames replay recorded material through the same loop instead.

Per tick model 1 sees the events of that second (hearing: an utterance that ENDED this second; sight: a caption when the
view changed) and outputs an action (WAIT/ATTEND/REPLY/THINK/INTERRUPT) and the memory gate (write this to memory?).
Actions are printed and logged; the thinker / memory are attached by the integration loop (experiments/integration),
this file is only the senses + model 1 and is meant to run for hours.  Ctrl+C stops it.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "model1"))
sys.path.insert(0, HERE)
import train_gru3 as tg  # noqa: E402
from asr import SR, Hearing, load_wav  # noqa: E402

ACTIONS = tg.ACTIONS


class MicStream:
    """ffmpeg dshow microphone -> 16 kHz mono float chunks (a thread fills a buffer, the loop drains it each second)"""

    def __init__(self, name):
        self.p = subprocess.Popen(["ffmpeg", "-v", "quiet", "-f", "dshow", "-i", f"audio={name}", "-f", "s16le", "-ac", "1",
                                   "-ar", str(SR), "-"], stdout=subprocess.PIPE)
        self.buf, self.lock = [], threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            b = self.p.stdout.read(SR * 2 // 4)
            if not b:
                break
            with self.lock:
                self.buf.append(np.frombuffer(b, np.int16).astype(np.float32) / 32768)

    def drain(self):
        with self.lock:
            out, self.buf = self.buf, []
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class CamStream:
    """ffmpeg dshow camera -> one 640x480 RGB frame per second"""

    def __init__(self, name):
        self.p = subprocess.Popen(["ffmpeg", "-v", "quiet", "-f", "dshow", "-i", f"video={name}", "-vf", "fps=1,scale=640:480",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)

    def frame(self):
        from PIL import Image
        b = self.p.stdout.read(640 * 480 * 3)
        return Image.frombytes("RGB", (640, 480), b) if len(b) == 640 * 480 * 3 else None


class Model1Live:
    """model 1 with a persistent hidden state and its own familiarity engram, one tick at a time"""

    def __init__(self, ckpt):
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer(tg.ENCODERS["qwen3emb"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
        tg.EMB = self.enc.get_sentence_embedding_dimension()
        self.eng = tg.SparseEngram(seed=1)
        self.m = tg.Model1(tg.EMB + 5).cuda()
        self.m.load_state_dict(torch.load(ckpt))
        self.m.eval()
        self.h = None
        self.busy_until, self.t = -1, 0

    def tick(self, event):
        x = np.zeros(tg.EMB + 5, np.float32)
        if event:
            v = self.enc.encode(event, normalize_embeddings=True)
            x[:tg.EMB] = v
            x[tg.EMB] = 1
            x[tg.EMB + 1], x[tg.EMB + 4] = self.eng.observe(torch.tensor(v, device="cuda", dtype=torch.float32), self.t)
        busy = self.t < self.busy_until
        x[tg.EMB + 2] = float(busy)
        x[tg.EMB + 3] = (self.t - (self.busy_until - tg.world2.THINK_TICKS)) / 25 if busy else 0
        with torch.no_grad():
            o, self.h = self.m(torch.tensor(x, device="cuda")[None, None], self.h)
        a = ACTIONS[int(o[0, 0, :5].argmax())]
        enc = bool(o[0, 0, 5] > 0)
        if a == "THINK":
            self.busy_until = self.t + tg.world2.THINK_TICKS
        elif a == "INTERRUPT":
            self.busy_until = self.t
        self.t += 1
        return a, enc, float(x[tg.EMB + 1])


def main(a):
    hearing = Hearing(a.asr_model) if (a.mic or a.wav) else None
    sight = None
    if a.cam or a.frames:
        from vision import Sight
        sight = Sight()
    mic = MicStream(a.mic) if a.mic else None
    cam = CamStream(a.cam) if a.cam else None
    wav = load_wav(a.wav) if a.wav else None
    frames = sorted(os.listdir(a.frames)) if a.frames else None
    m1 = Model1Live(a.ckpt)
    log = open(a.log, "a", encoding="utf-8")
    audio_acc, t0 = np.zeros(0, np.float32), time.time()
    print("alive; Ctrl+C to stop", flush=True)
    for tick in range(a.ticks if a.ticks > 0 else 10 ** 9):
        events = []
        # hearing: collect this second's audio; transcribe when a pause is likely (>= 0.6 s of quiet at the end)
        if mic or wav is not None:
            chunk = mic.drain() if mic else wav[tick * SR:(tick + 1) * SR]
            audio_acc = np.concatenate([audio_acc, chunk])
            tail = audio_acc[-int(0.6 * SR):]
            if len(audio_acc) >= SR and (len(tail) and float(np.abs(tail).mean()) < 0.01 or len(audio_acc) > 15 * SR):
                if float(np.abs(audio_acc).mean()) > 0.003:
                    for u in hearing.utterances(audio_acc):
                        events.append(u["event"])
                audio_acc = np.zeros(0, np.float32)
        # sight: one frame per second, caption only when the view changed
        if sight:
            from PIL import Image
            img = cam.frame() if cam else (Image.open(os.path.join(a.frames, frames[tick])) if tick < len(frames) else None)
            if img is not None:
                e = sight.step(tick, img)
                if e:
                    events.append(e["event"])
        ev = " | ".join(events) if events else None
        act, enc, fam = m1.tick(ev)
        rec = {"t": tick, "event": ev, "action": act, "encode": enc, "fam": round(fam, 3)}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        if ev or act != "WAIT":
            print(f"[{tick:5d}] {act:9s} enc={int(enc)} fam={fam:.2f} {ev or ''}", flush=True)
        if mic or cam:
            time.sleep(max(0.0, (t0 + tick + 1) - time.time()))   # real time: one tick per second
        elif wav is not None and tick * SR >= len(wav) and not events:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mic", help="dshow audio device name (opens the microphone)")
    p.add_argument("--cam", help="dshow video device name (opens the camera)")
    p.add_argument("--wav", help="replay a wav file instead of the microphone")
    p.add_argument("--frames", help="replay a folder of frames (one per second) instead of the camera")
    p.add_argument("--ckpt", default=os.path.join(ROOT, "model1", "results_w5", "w5_qwen3emb_fam_s0.pt"))
    p.add_argument("--asr_model", default="small")
    p.add_argument("--ticks", type=int, default=0, help="0 = run until stopped")
    p.add_argument("--log", default=os.path.join(HERE, "live_log.jsonl"))
    main(p.parse_args())
