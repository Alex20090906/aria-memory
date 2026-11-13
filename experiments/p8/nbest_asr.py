"""H39b (Gemini round 12): the 3 best recognition hypotheses for the heard segments around the script's questions.

faster-whisper's transcribe() returns only the best hypothesis; the CTranslate2 Whisper model underneath can return
several (beam search, num_hypotheses). For every heard segment that overlaps a question segment of the script (told
and never-told), the same model as the pipeline (large-v3-turbo, Chinese, the same initial prompt and the "Aria"
hotword prompt) decodes the segment's audio again with beam 5 and returns the 3 best texts with their scores.
CPU, int8 (this PC's GPU is used by another study).

    python nbest_asr.py <asr.json> <audio dir with timeline.json> <zh_scripts.json> <out.json>
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

SR = 16000
PROMPT = "以下是繁體中文的日常對話，助理的名字叫 Aria。"


def main(asr_path, audio_dir, sc_path, out_path):
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer
    m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8")
    tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language="zh")
    prev = tok.encode(" " + PROMPT)
    prompt = m.get_prompt(tok, prev, without_timestamps=True, prefix=None, hotwords="Aria")
    asr = {d["day"]: d for d in json.load(open(asr_path, encoding="utf-8"))}
    tl = {d["day"]: d for d in json.load(open(os.path.join(audio_dir, "timeline.json"), encoding="utf-8"))}
    sc = json.load(open(sc_path, encoding="utf-8"))
    out, t0, n = [], time.time(), 0
    for day in sorted(tl):
        qsegs = [g for g in tl[day]["segments"] if any(q["day"] == day and q["text"] == g["text"] for q in sc["questions"])]
        if not qsegs:
            continue
        audio, _ = sf.read(os.path.join(audio_dir, tl[day]["wav"]), dtype="float32")
        for i, h in enumerate(asr[day]["heard"]):
            if not any(min(h["end"], g["end"]) - max(h["start"], g["start"]) > 0.3 for g in qsegs):
                continue
            seg = audio[int(h["start"] * SR):int(h["end"] * SR)]
            feats = m.feature_extractor(seg)
            frames = feats.shape[-1]
            feats = np.pad(feats, ((0, 0), (0, max(0, 3000 - frames))))[:, :3000]
            enc = m.encode(feats)
            res = m.model.generate(enc, [prompt], beam_size=5, num_hypotheses=3, return_scores=True,
                                   max_length=224, suppress_blank=True, suppress_tokens=[-1])[0]
            hyps = [{"text": tok.decode([t for t in ids if t < tok.eot]).strip(), "score": round(float(s), 3)}
                    for ids, s in zip(res.sequences_ids, res.scores)]
            out.append({"day": day, "i": i, "start": h["start"], "end": h["end"], "owner": h["owner"],
                        "best_in_run": h["text"], "nbest": hyps})
            n += 1
            if n % 10 == 0:
                print(f"{n} segments, {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done", n, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:5])
