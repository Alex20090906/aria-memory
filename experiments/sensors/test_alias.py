"""Synthetic household sessions for the alias map: 3-4 people with hidden voice ids talk for a day.

Each session: people get random speaker#N ids; lines are casual chatter, vocatives ("Leo, did you feed the cat?")
followed (usually) by the addressee's reply, occasional self-introductions (guests only), and fact lines ("my keys are
in the glovebox") which must end up attributed to the right person. Noise: replies sometimes come from a third
person (20%), names mentioned non-vocatively ("Leo said it's fine"), the assistant's name.
Metrics: alias accuracy (assigned names that are right), coverage (speakers that got a name), and
fact attribution = fact lines rewritten with the true owner name, at the moment they are spoken.
"""
import random
import statistics

from alias import AliasMap

PEOPLE = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Ken", "Lena", "Ivan"]
CHAT = ["ugh it's raining again", "did anyone see my charger", "dinner at seven?", "the wifi is dead lol",
        "I'm so tired today", "who ate the last yogurt", "we should clean the garage this weekend",
        "that show last night was wild", "can you turn the heater down", "I'll be back late"]
ASKS = ["did you feed the cat?", "are you coming tonight?", "can you grab milk?", "where were you?",
        "you want tea?", "is the car free tomorrow?"]
REPLIES = ["yeah sure", "nah not yet", "in a minute", "already did", "maybe, why?", "ok ok"]
FACTS = ["my keys are in the glovebox btw", "I left my watch on the windowsill", "my passport's in the freezer, don't ask",
         "put my glasses behind the mirror", "my wallet is under the doormat lol"]


def session(rng, n_lines=200):
    people = rng.sample(PEOPLE, rng.choice([3, 4]))
    ids = {p: f"speaker#{i + 1}" for i, p in enumerate(rng.sample(people, len(people)))}
    guest = people[-1] if rng.random() < 0.5 else None
    lines, t = [], 0.0
    if guest:
        lines.append((t, guest, rng.choice([f"hi, I'm {guest}", f"hey all, {guest} here", f"it's {guest}, sorry I'm late"])))
    while len(lines) < n_lines:
        t += rng.uniform(2, 30)
        a = rng.choice(people)
        r = rng.random()
        if r < 0.35:
            b = rng.choice([p for p in people if p != a])
            q = rng.choice([f"{b}, {rng.choice(ASKS)}", f"hey {b}, {rng.choice(ASKS)}", f"{rng.choice(ASKS)[:-1]}, {b}?"])
            lines.append((t, a, q))
            t += rng.uniform(1, 6)
            replier = b if rng.random() < 0.8 else rng.choice([p for p in people if p != a])
            lines.append((t, replier, rng.choice(REPLIES)))
        elif r < 0.45:
            other = rng.choice([p for p in people if p != a])
            lines.append((t, a, f"{other} said it's fine"))
        elif r < 0.55:
            lines.append((t, a, "Aria, " + rng.choice(["what's the weather", "set a timer", "play some music"])))
        elif r < 0.65:
            lines.append((t, a, rng.choice(FACTS)))
        else:
            lines.append((t, a, rng.choice(CHAT)))
    return people, ids, lines


def run(seed):
    rng = random.Random(seed)
    people, ids, lines = session(rng)
    am = AliasMap()
    fact_ok = fact_n = 0
    released = []
    first_named = {}
    for i, (t, p, text) in enumerate(lines):
        sid = ids[p]
        am.observe(t, sid, text)
        if any(text.startswith(x[:12]) for x in FACTS):
            fact_n += 1
            fact_ok += am.rewrite(sid, text).startswith(p + ":")
            released += [(p, r) for r in am.hold_or_release(sid, text)]
        else:
            released += [(None, r) for r in am.hold_or_release(sid, None)]
        if am.name(sid) and sid not in first_named:
            first_named[sid] = i
    tab = am.table()
    # held lines released later carry no true owner in `released` when flushed by a non-fact line: check by content
    owners = {}
    for (t_, p_, x_) in lines:
        if any(x_.startswith(f[:12]) for f in FACTS):
            owners.setdefault(x_, set()).add(p_)
    rel_ok = sum(1 for _, r in released if r.split(": ", 1)[0] in owners.get(r.split(": ", 1)[1], set()))
    held_left = sum(len(v) for v in getattr(am, "held", {}).values())
    inv = {v: k for k, v in ids.items()}
    named = {s: n for s, n in tab.items() if n}
    right = sum(1 for s, n in named.items() if inv[s] == n)
    return {"acc": right / max(1, len(named)), "cover": len(named) / len(people), "fact": fact_ok / max(1, fact_n),
            "wrong": len(named) - right, "eventual": rel_ok / max(1, fact_n), "unreleased": held_left / max(1, fact_n), "lines_to_name": statistics.mean(first_named.values()) if first_named else None}


if __name__ == "__main__":
    rs = [run(s) for s in range(50)]
    for k in ("acc", "cover", "fact", "eventual", "unreleased", "wrong"):
        print(k, round(statistics.mean(r[k] for r in rs), 3))
    print("mean lines until a speaker gets a name", round(statistics.mean(r["lines_to_name"] for r in rs if r["lines_to_name"] is not None), 1))
    print("sessions with any wrong name", sum(1 for r in rs if r["wrong"]), "/ 50")
