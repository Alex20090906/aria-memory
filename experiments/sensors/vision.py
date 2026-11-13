"""Sight for model 1: camera frames -> "[CAM] ..." events, only when the view changes.

Cheap change detection runs on every frame (downscaled grey difference); a caption is computed only when the
change is large enough, so a still room costs almost nothing (the habituation itself is model 1's job, via the
familiarity signal; this only avoids describing identical frames).

Captioner: a small vision-language model through transformers (default Qwen/Qwen2-VL-2B-Instruct), asked for one
short present-tense sentence about what changed or what is happening.
Sources: a folder/video of frames for tests, or a camera through ffmpeg (dshow), only started on request.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image

PROMPT = ("Describe in one short present-tense sentence what is happening in this camera view. "
          "Mention people, animals and anything unusual. No preamble.")


class Sight:
    def __init__(self, model="Salesforce/blip-image-captioning-large", device="cuda", thresh=0.01):
        # BLIP: short scene captions, no torchvision dependency (Qwen2-VL needs a matching torchvision build)
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.proc = BlipProcessor.from_pretrained(model)
        self.m = BlipForConditionalGeneration.from_pretrained(model, dtype=torch.float16).to(device).eval()
        self.device = device
        self.prev = None
        self.thresh = thresh

    def changed(self, img):
        # fraction of pixels that changed clearly: a small new object still counts, sensor noise does not
        g = np.asarray(img.convert("L").resize((128, 96)), np.float32) / 255
        d = 1.0 if self.prev is None else float((np.abs(g - self.prev) > 0.12).mean())
        self.prev = g
        return d >= self.thresh, d

    def caption(self, img):
        import torch
        inp = self.proc(images=img.convert("RGB"), return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            out = self.m.generate(**inp, max_new_tokens=25, num_beams=3)
        return self.proc.decode(out[0], skip_special_tokens=True).strip()

    def step(self, t, img):
        ch, d = self.changed(img)
        if not ch:
            return None
        return {"t": t, "diff": round(d, 3), "event": f"[CAM] {self.caption(img)}"}


def frames_from_video(path, fps=1):
    """one frame per second from a video file (tests)"""
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-vf", f"fps={fps},scale=640:-1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    from io import BytesIO
    i, t = 0, 0
    while True:
        j = out.find(b"\x89PNG", i + 1)
        chunk = out[i:j if j > 0 else None]
        if chunk:
            yield t, Image.open(BytesIO(chunk))
            t += 1
        if j < 0:
            break
        i = j


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="a video file or a folder of images (sorted by name, one per second)")
    a = p.parse_args()
    s = Sight()
    if os.path.isdir(a.source):
        it = ((t, Image.open(os.path.join(a.source, f))) for t, f in enumerate(sorted(os.listdir(a.source))))
    else:
        it = frames_from_video(a.source)
    for t, img in it:
        e = s.step(t, img)
        if e:
            print(json.dumps(e, ensure_ascii=False), flush=True)
