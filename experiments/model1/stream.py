"""Synthetic sensory stream for the always-on consciousness model (model 1).

One tick = one second. Each tick has zero or one sensor event. Every event carries the
ideal action, so we can score model 1. Ground truth depends on context:
  - a novel scene deserves ATTEND the first time, WAIT when it repeats (habituation)
  - "never mind" deserves INTERRUPT only while the thinker is busy
The thinker is simulated: THINK makes it busy for THINK_TICKS, then a result event arrives.
"""
import random

ACTIONS = ["WAIT", "ATTEND", "REPLY", "THINK", "INTERRUPT"]
THINK_TICKS = 25
HABITUATION_WINDOW = 400  # a scene seen within this many ticks no longer deserves ATTEND

AMBIENT = [
    "[MIC] a fan is humming",
    "[MIC] traffic noise outside",
    "[CAM] the room looks unchanged",
    "[MIC] a clock ticks",
]
TV = [
    "[MIC] TV: tomorrow will be cloudy with light rain in the north",
    "[MIC] TV: the stock market closed higher today",
    "[MIC] TV: and now, a word from our sponsors",
    "[MIC] TV: scientists discovered a new species of frog",
    "[MIC] TV: traffic is heavy on the main highway",
]
SCENES = [
    "[CAM] a cat walks into the room",
    "[CAM] the lights were switched off",
    "[CAM] a delivery box is on the table",
    "[CAM] someone opened the window",
    "[CAM] a dog is barking at the door",
    "[CAM] a person entered the room",
]
GREETINGS = [
    "[MIC] Axel: hi Aria, good morning",
    "[MIC] Axel: hey Aria, are you there?",
    "[MIC] Axel: thanks Aria, that helps",
    "[MIC] Axel: Aria, what is your name again?",
    "[MIC] Axel: good night Aria",
]
COMPLEX = [
    "[MIC] Axel: Aria, can you plan a five day trip to Japan for me?",
    "[MIC] Axel: Aria, help me debug why my Python script crashes on startup",
    "[MIC] Axel: Aria, compare three ways to save money on groceries and pick the best",
    "[MIC] Axel: Aria, write a short story about a lighthouse keeper",
    "[MIC] Axel: Aria, explain how vaccines train the immune system, step by step",
]
CANCEL = "[MIC] Axel: never mind Aria, stop working on that"
RESULT = "[THINKER] result is ready"


def generate(n_ticks=3000, seed=0):
    rng = random.Random(seed)
    ticks = []
    busy_until = -1
    last_seen_scene = {}
    pending_result = None
    for t in range(n_ticks):
        ev, ideal, kind = None, "WAIT", "empty"
        if pending_result is not None and t == pending_result:
            ev, ideal, kind = RESULT, "REPLY", "result"
            pending_result = None
        else:
            r = rng.random()
            thinker_busy = t < busy_until
            if r < 0.25:
                ev, kind = rng.choice(AMBIENT), "ambient"
            elif r < 0.31:
                ev, kind = rng.choice(TV), "tv"
            elif r < 0.335:
                ev, kind = rng.choice(SCENES), "scene"
                seen = last_seen_scene.get(ev)
                ideal = "ATTEND" if seen is None or t - seen > HABITUATION_WINDOW else "WAIT"
                if ideal == "WAIT":
                    kind = "scene_repeat"
                last_seen_scene[ev] = t
            elif r < 0.345:
                ev, ideal, kind = rng.choice(GREETINGS), "REPLY", "greeting"
            elif r < 0.352 and not thinker_busy:
                ev, ideal, kind = rng.choice(COMPLEX), "THINK", "complex"
                busy_until = t + THINK_TICKS
                pending_result = busy_until
            elif r < 0.356 and thinker_busy:
                ev, ideal, kind = CANCEL, "INTERRUPT", "cancel"
                busy_until = t
                pending_result = None
        ticks.append({"t": t, "event": ev, "ideal": ideal, "kind": kind,
                      "thinker_busy": t < busy_until})
    return ticks


if __name__ == "__main__":
    from collections import Counter
    ts = generate()
    print(Counter(x["kind"] for x in ts))
    print(Counter(x["ideal"] for x in ts))
