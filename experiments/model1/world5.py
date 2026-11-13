"""World v5 = world v4 as a real microphone hears it: the speaker label is gone.

A microphone with speech recognition gives text, not "TV:" or "Leo:". Every [MIC] event becomes
"[MIC] someone: <words>", so model 1 has to tell the TV, people talking to each other and people talking to Aria
apart from the words alone (names inside the words, "Aria, ..." addressing, broadcast style).
"""
import re

import world4
from world2 import ACTIONS  # noqa: F401


def anonymize(ev):
    if ev is None or not ev.startswith("[MIC] "):
        return ev
    body = ev[6:]
    m = re.match(r"^([^:]{1,40}):\s(.*)$", body)
    if m and not body.startswith("someone"):
        body = m.group(2)            # "Leo: Aria, ..." -> "Aria, ..."; "TV: the market ..." -> "the market ..."
    return f"[MIC] someone: {body}"


def generate(n_ticks, seed, part, **kw):
    ticks = world4.generate(n_ticks, seed, part, **kw)
    for x in ticks:
        x["event"] = anonymize(x["event"])
    return ticks


if __name__ == "__main__":
    t = generate(3000, 0, "test")
    for x in t:
        if x["event"] and x["kind"] in ("background", "overheard", "overheard_fact", "inform", "greeting"):
            print(x["kind"], "|", x["event"])
            if sum(1 for _ in range(1)) and x["t"] > 400:
                break
