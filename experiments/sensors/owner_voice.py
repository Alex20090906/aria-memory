"""Single-user mode: is this utterance the owner's voice?

The owner enrolls with a few utterances; each later utterance gets a resemblyzer embedding and a cosine to the
owner centroid. >= accept: owner (model 1 may act, memory owner of "my/I" = the owner). < reject: someone else
(background: no action, no write). In between: unsure (treated as not-owner for writes, model 1 may still attend).
The centroid adapts slowly on confident owner utterances (voice drift over months: colds, mood, microphone).
"""
import numpy as np

SR = 16000


class OwnerVoice:
    def __init__(self, accept=0.80, reject=0.70, adapt=0.02):
        from resemblyzer import VoiceEncoder
        self.enc = VoiceEncoder("cpu", verbose=False)
        self.c = None
        self.accept, self.reject, self.adapt = accept, reject, adapt

    def embed(self, wav16k):
        return self.enc.embed_utterance(wav16k.astype(np.float32))

    def enroll(self, wavs):
        es = [self.embed(w) for w in wavs]
        c = np.mean(es, 0)
        self.c = c / np.linalg.norm(c)
        return [float(e @ self.c) for e in es]

    def score(self, wav16k):
        if len(wav16k) < SR // 2:
            return None
        return float(self.embed(wav16k) @ self.c)

    def verify(self, wav16k):
        s = self.score(wav16k)
        if s is None:
            return "unsure", s
        if s >= self.accept:
            e = self.embed(wav16k)
            c = (1 - self.adapt) * self.c + self.adapt * e
            self.c = c / np.linalg.norm(c)
            return "owner", s
        return ("other" if s < self.reject else "unsure"), s
