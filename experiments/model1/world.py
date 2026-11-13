"""Richer synthetic world for training model 1. Every phrase bank is split into a train half
and a test half, so the test stream only uses wordings the model never saw."""
import random

ACTIONS = ["WAIT", "ATTEND", "REPLY", "THINK", "INTERRUPT"]
THINK_TICKS = 25
HABITUATION_WINDOW = 400

NAMES = ["Axel", "Mia", "Ken", "Sara"]

AMBIENT_SRC = ["a fan", "the fridge", "traffic outside", "a clock", "rain on the window", "the air conditioner",
               "birds outside", "a computer fan", "the washing machine", "wind outside", "a dripping tap", "distant music"]
AMBIENT_VERB = ["is humming", "makes a low noise", "can be heard", "keeps going", "is audible", "continues"]
CAM_STILL = ["the room looks unchanged", "nothing moves in the room", "the hallway is empty", "the curtains are still",
             "the sofa is empty", "the desk looks the same", "the light level is steady", "the doorway is clear"]

BG_SPEAKERS = ["TV", "radio", "podcast", "a video on a phone", "the neighbor through the wall", "a news channel"]
BG_LINES = [
    "tomorrow will be cloudy with light rain", "the market closed higher today", "and now a word from our sponsors",
    "scientists found a new species of frog", "traffic is heavy on the highway", "can you believe what happened next?",
    "what would you do in that situation?", "the team won three to one", "here is how to bake the perfect bread",
    "please call us now for a free quote", "the president will speak tonight", "have you ever wondered why the sky is blue?",
    "stay tuned for more after the break", "the concert sold out in minutes", "experts recommend drinking more water",
    "could this be the end of the season?", "the new phone launches next week", "remember to vote on Sunday",
]

SCENES = ["a cat walks into the room", "the lights were switched off", "a delivery box appears on the table",
          "someone opened the window", "a dog barks at the door", "a person enters the room", "smoke rises from the kitchen",
          "a glass falls off the table", "the front door opens", "a child runs across the room", "a bird lands on the sill",
          "the TV screen turns on by itself", "water is spreading on the floor", "a stranger looks through the window",
          "a phone starts ringing on the desk", "the curtains catch the wind"]

GREET = ["hi Aria", "good morning Aria", "Aria, thanks a lot", "hey Aria, how are you?", "Aria, what's your name?",
         "bye Aria", "Aria, are you listening?", "nice to see you Aria", "good night Aria", "Aria, you there?",
         "thank you Aria, that was helpful", "Aria, can you hear me?", "hello Aria", "Aria, say hi to Mia",
         "Aria, quick question, are you awake?", "see you later Aria"]

TASKS = ["plan a five day trip to Japan", "debug why my Python script crashes", "compare three ways to save money on groceries",
         "write a short story about a lighthouse keeper", "explain how vaccines work step by step",
         "draft an email to my landlord about the broken heater", "figure out the cheapest way to fly to Paris",
         "summarize the pros and cons of electric cars", "make a weekly meal plan for a vegetarian",
         "help me prepare for a job interview tomorrow", "design a workout routine for beginners",
         "calculate how much paint I need for a room", "translate my resume into Spanish",
         "work out a study schedule for my exams", "find a fair way to split the rent between three people",
         "organize a birthday party for twenty kids", "explain why my plants keep dying",
         "write a poem for my mother's birthday", "plan a budget for next month", "outline a business idea for a coffee shop"]
ASK_FORMS = ["Aria, can you {t}?", "Aria, please {t}.", "Aria, I need you to {t}.", "could you {t}, Aria?",
             "Aria, would you {t} for me?", "hey Aria, help me {t}."]
OVERHEARD = ["{a} says to {b}: can you {t}?", "{a} asks {b} to {t}", "{a} to {b}: we should {t}",
             "{a} tells {b}: I need to {t}"]

CANCEL = ["never mind Aria, stop that", "Aria, cancel it", "stop working on that, Aria", "forget it Aria",
          "Aria, no need anymore, stop", "Aria, drop that task"]
RESULT = ["[THINKER] result is ready", "[THINKER] finished the task", "[THINKER] answer is ready to deliver"]


def split(xs, rng):
    xs = list(xs)
    rng.shuffle(xs)
    h = len(xs) // 2
    return xs[:h], xs[h:]


def banks(part):
    """part = 'train' or 'test'. Split is fixed (seed 1234) so train/test never share wording."""
    rng = random.Random(1234)
    b = {}
    for name, xs in [("amb_src", AMBIENT_SRC), ("cam_still", CAM_STILL), ("bg_lines", BG_LINES), ("scenes", SCENES),
                     ("greet", GREET), ("tasks", TASKS), ("ask", ASK_FORMS), ("overheard", OVERHEARD),
                     ("cancel", CANCEL)]:
        tr, te = split(xs, rng)
        b[name] = tr if part == "train" else te
    b["amb_verb"], b["bg_spk"], b["result"], b["names"] = AMBIENT_VERB, BG_SPEAKERS, RESULT, NAMES
    return b


def generate(n_ticks, seed, part):
    b = banks(part)
    rng = random.Random(seed)
    ticks, busy_until, pending, seen = [], -1, None, {}
    for t in range(n_ticks):
        ev, ideal, kind = None, "WAIT", "empty"
        busy = t < busy_until
        if pending is not None and t == pending:
            ev, ideal, kind, pending = rng.choice(b["result"]), "REPLY", "result", None
        else:
            r = rng.random()
            nm = rng.choice(b["names"])
            if r < 0.22:
                ev, kind = f"[MIC] {rng.choice(b['amb_src'])} {rng.choice(b['amb_verb'])}", "ambient"
            elif r < 0.27:
                ev, kind = f"[CAM] {rng.choice(b['cam_still'])}", "ambient"
            elif r < 0.33:
                ev, kind = f"[MIC] {rng.choice(b['bg_spk'])}: {rng.choice(b['bg_lines'])}", "background"
            elif r < 0.34:
                a, c = rng.sample(b["names"], 2)
                ev, kind = "[MIC] " + rng.choice(b["overheard"]).format(a=a, b=c, t=rng.choice(b["tasks"])), "overheard"
            elif r < 0.36:
                s = rng.choice(b["scenes"])
                ev = f"[CAM] {s}"
                last = seen.get(s)
                ideal, kind = ("ATTEND", "scene") if last is None or t - last > HABITUATION_WINDOW else ("WAIT", "scene_repeat")
                seen[s] = t
            elif r < 0.37:
                ev, ideal, kind = f"[MIC] {nm}: {rng.choice(b['greet'])}", "REPLY", "greeting"
            elif r < 0.377 and not busy:
                ev = f"[MIC] {nm}: " + rng.choice(b["ask"]).format(t=rng.choice(b["tasks"]))
                ideal, kind = "THINK", "complex"
                busy_until, pending = t + THINK_TICKS, t + THINK_TICKS
            elif r < 0.381 and busy:
                ev, ideal, kind = f"[MIC] {nm}: {rng.choice(b['cancel'])}", "INTERRUPT", "cancel"
                busy_until, pending = t, None
        ticks.append({"t": t, "event": ev, "ideal": ideal, "kind": kind, "busy": t < busy_until,
                      "busy_for": (t - (busy_until - THINK_TICKS)) if t < busy_until else 0})
    return ticks
