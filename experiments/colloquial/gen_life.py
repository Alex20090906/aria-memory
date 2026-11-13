"""Single-user life data: one person talks to Aria for a year (casual English, fillers, typos, indirect statements).

Memory schema (Gemini round 7): [entity|slot] -> value, slot from a fixed set. Facts change over time (updates).
Output life.json:
  events:    {"day", "say", "entity", "slot", "value", "update_of": index of the event it replaces or null}
  questions: {"day", "ask", "entity", "slot", "expect": value valid on that day or null (never told),
              "stale": older value if the fact was updated, else null}
  chatter:   lines with no fact
Auth: GOOGLE_APPLICATION_CREDENTIALS (Vertex AI service account, never copied).
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "discussions"))
import gemini_discuss as gd  # noqa: E402

SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]

PROMPT = """Write test data for a home assistant called Aria that lives with ONE person (call them "me" in the data; never
give the person a name). Over a year the person mentions facts about their life to Aria, casually, the way people
talk to a housemate: fillers, contractions, half-sentences, the odd typo, jokes, sometimes only implying the fact.

Memory schema: each fact is [entity|slot] -> value, where slot is one of {slots}:
  location (where a thing is), time (when something happens: appointments, deadlines), status (state of a thing:
  broken, lent out, expired), preference (what the person likes/dislikes), plan (what they intend to do),
  relation (who someone is to them), attribute (a property: color, size, code-free details), value (a number or
  amount: shoe size, budget). The entity is a short lowercase noun phrase ("passport", "dentist", "sister maya",
  "coffee", "car"). The value is a short phrase as it should be recalled ("in the freezer", "friday at 3pm",
  "oat milk, no sugar", "college roommate").

Rules:
- {n_entities} different entities; {n_events} fact events in total, spread over days 1..365 in order.
- About {n_updates} events UPDATE an earlier fact (same entity and slot, new value, e.g. the passport moved, the
  dentist got rescheduled). Mark them with "update_of" = index of the replaced event, else null.
- No secrets: no passwords, PINs, card numbers, addresses, phone numbers.
- {n_questions} questions spread over the year, each asked AFTER the fact it is about, casual wording, sometimes
  terse ("passport??"), sometimes "did I ever tell you...". "expect" = the value valid on that day; if the fact was
  updated before that day, "expect" is the new value and "stale" the old one. Also {n_never} questions about facts
  that were NEVER told (expect null).
- {n_chatter} chatter lines with no facts (weather, food, TV, mood).

Return ONLY JSON: {{"events": [{{"day": int, "say": str, "entity": str, "slot": str, "value": str,
"update_of": int|null}}], "questions": [{{"day": int, "ask": str, "entity": str, "slot": str, "expect": str|null,
"stale": str|null}}], "chatter": [{{"day": int, "say": str}}]}}"""


def main(seed=0):
    p = PROMPT.format(slots=", ".join(SLOTS), n_entities=45, n_events=90, n_updates=18, n_questions=70, n_never=12,
                      n_chatter=60)
    for attempt in range(3):
        out = gd.call([{"role": "user", "parts": [{"text": p}]}]).strip()
        if out.startswith("```"):
            out = out.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            d = json.loads(out)
            break
        except Exception as e:
            print("retry", e, flush=True)
    else:
        raise SystemExit("no JSON")
    bad = [e for e in d["events"] if e["slot"] not in SLOTS]
    print("events", len(d["events"]), "updates", sum(1 for e in d["events"] if e.get("update_of") is not None),
          "questions", len(d["questions"]), "never", sum(1 for q in d["questions"] if q["expect"] is None),
          "stale-tests", sum(1 for q in d["questions"] if q.get("stale")), "chatter", len(d["chatter"]),
          "bad slots", len(bad))
    json.dump(d, open(os.path.join(HERE, "life.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
