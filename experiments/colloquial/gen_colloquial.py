"""Colloquial test data (expansion plan, step "直觀程度"): the same kind of facts, said the way people talk.

For each fact (owner, object, place) Gemini writes
  tell_direct    casual telling with fillers ("uh, so I kinda shoved my passport in the freezer, remember that")
  tell_indirect  the fact only implied ("don't open the freezer yet, my passport's chilling in there")
  ask_casual     casual question by the owner ("hey where'd my passport end up again")
  ask_terse      very short ("passport??")
  ask_meta       asks whether they told Aria ("did I ever tell you where my passport is?")
  ask_other      someone else asks about the owner's thing ("Aria, any clue where Leo put his passport?") + asker name
plus distractor chatter lines with no fact. Exact answers stay with the fact so scoring is automatic.
Writes colloquial.json. Auth: GOOGLE_APPLICATION_CREDENTIALS (Vertex AI service account, never copied).
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "discussions"))
import gemini_discuss as gd  # noqa: E402

NAMES = ["Leo", "Mia", "Omar", "Sara", "Nina", "Tom", "Yumi", "Axel", "Ken", "Lena", "Ivan", "Rosa", "Pablo", "Hana",
         "Ben", "Chloe", "Dan", "Emma", "Finn", "Grace"]
OBJS = ["passport", "spare key", "watch", "glasses", "ticket", "charger", "ring", "wallet", "umbrella", "headphones",
        "notebook", "camera", "scarf", "medicine", "bank card", "laptop", "keys", "hat", "tablet", "sunglasses"]
PLACES = ["in the red teapot", "in the blue envelope", "in the garage drawer", "in the freezer", "in the toolbox",
          "in the rice jar", "in the old boot", "in the piano bench", "in the laundry basket", "in the glovebox",
          "under the doormat", "under the piano", "behind the bookshelf", "behind the mirror", "on the windowsill",
          "on top of the fridge"]

PROMPT = """You write test utterances for a home assistant called Aria. People talk to Aria the way they talk to a
housemate: casual, sloppy, fillers, contractions, half-sentences, jokes, sometimes a typo. Do NOT write clean textbook
sentences. Keep the object word recognisable (the word "{obj}" or an obvious plural/casual form of it) and keep the
place phrase recognisable (its main noun must appear), but you may wrap them in casual wording.

Fact: {owner}'s {obj} is {place}.  Another person present: {other}.

Return ONLY a JSON object with these string fields:
"tell_direct": {owner} tells Aria the fact casually, 1-2 sentences, with filler words.
"tell_indirect": {owner} says something to Aria or to {other} that only IMPLIES the fact (e.g. a warning, an excuse,
   a joke), never a plain "I put X in Y". Aria must still be able to infer it.
"ask_casual": days later {owner} asks Aria where it is, casually.
"ask_terse": {owner} asks in 1-4 words.
"ask_meta": {owner} asks whether they ever told Aria where it is.
"ask_other": {other} asks Aria where {owner}'s {obj} is (mention {owner} by name).
"""

CHATTER_PROMPT = """Write {n} different lines of everyday chatter that people at home say near a home assistant called
Aria: complaints, jokes, plans, TV comments, questions about weather or food. None of them may state where any object
is. Casual, sloppy, realistic. Return ONLY a JSON list of strings, each formatted "Name: text" using names from
{names}."""


def ask_json(prompt):
    out = gd.call([{"role": "user", "parts": [{"text": prompt}]}]).strip()
    if out.startswith("```"):
        out = out.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(out)


def main(n=40, seed=0):
    rng = random.Random(seed)
    pairs = [(a, o) for a in NAMES for o in OBJS]
    rng.shuffle(pairs)
    facts = []
    for owner, obj in pairs[:n]:
        place = rng.choice(PLACES)
        other = rng.choice([x for x in NAMES if x != owner])
        for attempt in range(3):
            try:
                u = ask_json(PROMPT.format(owner=owner, obj=obj, place=place, other=other))
                break
            except Exception as e:
                print("retry", owner, obj, e, flush=True)
        else:
            continue
        facts.append({"owner": owner, "object": obj, "place": place, "other": other, **u})
        print(len(facts), owner, obj, "|", u.get("tell_indirect", "")[:80], flush=True)
    chatter = ask_json(CHATTER_PROMPT.format(n=60, names=", ".join(NAMES)))
    json.dump({"facts": facts, "chatter": chatter}, open(os.path.join(HERE, "colloquial.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("facts", len(facts), "chatter", len(chatter))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
