"""World v2 for model 1: much larger, compositional phrase banks.

Changes from world.py:
- Every bank is 3-6x larger, and scenes are compositional (subject x action), so the test
  stream uses subjects, actions and sentences that never appear in training.
- Repeats are scheduled explicitly. A scene is either brand new, a repeat of something seen
  20-250 ticks ago (should be ignored), or a return after more than RECOVER ticks (should be
  noticed again). Gaps in between are never generated, so the label is not ambiguous at the
  habituation boundary (in world.py most "new" scenes were returns just past 400 ticks).
- More complex requests, overheard requests and simple questions (to REPLY), so the
  addressed / not addressed / simple / complex distinctions have enough examples.
"""
import random

ACTIONS = ["WAIT", "ATTEND", "REPLY", "THINK", "INTERRUPT"]
THINK_TICKS = 25
REPEAT_MAX = 250
RECOVER = 1500

NAMES = ["Axel", "Mia", "Ken", "Sara", "Leo", "Nina", "Tom", "Yumi", "Omar", "Lena"]

AMBIENT_SRC = ["a fan", "the fridge", "traffic outside", "a clock", "rain on the window", "the air conditioner",
               "birds outside", "a computer fan", "the washing machine", "wind outside", "a dripping tap",
               "distant music", "the heater", "a lawn mower far away", "the dishwasher", "crickets outside",
               "a car alarm far away", "construction noise down the street", "the water pipes", "a ceiling fan",
               "the aquarium pump", "footsteps upstairs", "a dog barking far away", "the elevator in the hallway",
               "people chatting in the street", "a train passing in the distance", "the radiator", "the printer",
               "a neighbor's TV through the wall", "the wind chimes"]
AMBIENT_VERB = ["is humming", "makes a low noise", "can be heard", "keeps going", "is audible", "continues",
                "goes on in the background", "is faintly heard"]
CAM_STILL = ["the room looks unchanged", "nothing moves in the room", "the hallway is empty", "the curtains are still",
             "the sofa is empty", "the desk looks the same", "the light level is steady", "the doorway is clear",
             "the kitchen is quiet", "the bookshelf is undisturbed", "the window shows the same street",
             "the floor is clear", "the plants are where they were", "the table is as before", "no one is in view",
             "the lamp is still on", "the corridor stays dark", "the chair has not moved", "the screen is still off",
             "the shadows shift slightly as clouds pass", "the clock on the wall ticks", "dust floats in the light",
             "the bed is made", "the bathroom door stays closed"]

BG_SPEAKERS = ["TV", "radio", "podcast", "a video on a phone", "the neighbor through the wall", "a news channel",
               "a song on the speaker", "an advert on the tablet", "a cartoon on the TV", "a voice message playing"]
BG_LINES = [
    "tomorrow will be cloudy with light rain", "the market closed higher today", "and now a word from our sponsors",
    "scientists found a new species of frog", "traffic is heavy on the highway", "can you believe what happened next?",
    "what would you do in that situation?", "the team won three to one", "here is how to bake the perfect bread",
    "please call us now for a free quote", "the president will speak tonight", "have you ever wondered why the sky is blue?",
    "stay tuned for more after the break", "the concert sold out in minutes", "experts recommend drinking more water",
    "could this be the end of the season?", "the new phone launches next week", "remember to vote on Sunday",
    "can you help me find my keys?", "tell me, where were you last night?", "hey, are you listening to me?",
    "please explain how this machine works", "the recipe needs two cups of flour", "you won't believe this deal",
    "let's plan the perfect holiday together", "the storm is moving east", "click the link below to subscribe",
    "hello and welcome back to the show", "thanks for watching, see you tomorrow", "the museum opens a new exhibit",
    "could you write down the number?", "the singer announced a world tour", "water levels rose overnight",
    "what is the meaning of life?", "and the winner is", "this product changed my life",
]

SCENE_SUBJ = ["a cat", "a dog", "a man in a red coat", "a woman with an umbrella", "a small child", "the neighbor",
              "a delivery driver", "a bird", "an elderly man", "a teenager on a skateboard", "a police officer",
              "a stranger", "a mouse", "a squirrel", "a woman in a blue dress", "a man carrying boxes",
              "two children", "a large dog", "a girl with a balloon", "the cleaner", "a cyclist", "a postman",
              "a man in a hoodie", "a toddler"]
SCENE_ACT = ["walks into the room", "knocks on the window", "runs across the hallway", "stands at the door",
             "jumps onto the table", "looks into the camera", "drops something on the floor", "opens the fridge",
             "sits down on the sofa", "climbs onto the shelf", "waves at the camera", "leaves through the back door",
             "picks up a bag from the floor", "peeks through the curtains", "falls over near the stairs",
             "starts crying in the corner", "carries a ladder past the window", "rings the doorbell",
             "moves the chair", "turns on the lamp", "hides behind the sofa", "lies down on the floor",
             "takes something from the desk", "stares out of the window"]
SCENE_EVENT = ["smoke rises from the kitchen", "the lights were switched off", "a delivery box appears on the table",
               "the window was opened", "a glass falls off the table", "the front door opens",
               "the TV screen turns on by itself", "water is spreading on the floor", "a phone starts ringing on the desk",
               "the curtains catch the wind", "a picture falls off the wall", "sparks come from the socket",
               "the ceiling light flickers", "a vase breaks on the floor", "the oven door is left open",
               "a red light blinks on the router", "the bookshelf tips over", "snow starts falling outside",
               "a fire alarm light turns on", "a ball rolls across the floor", "the garage door goes up",
               "a car pulls into the driveway", "the power goes out", "a shadow passes the window",
               "the tap is left running", "an umbrella falls open", "the cat bowl is knocked over",
               "the blinds roll down", "a package is left at the door", "fog covers the window"]

GREET = ["hi Aria", "good morning Aria", "Aria, thanks a lot", "hey Aria, how are you?", "Aria, what's your name?",
         "bye Aria", "Aria, are you listening?", "nice to see you Aria", "good night Aria", "Aria, you there?",
         "thank you Aria, that was helpful", "Aria, can you hear me?", "hello Aria", "Aria, say hi to Mia",
         "Aria, quick question, are you awake?", "see you later Aria", "morning Aria", "Aria, I'm home",
         "Aria, I'm leaving now", "great job Aria", "Aria, how's it going?", "evening Aria", "Aria, are you okay?",
         "thanks Aria", "Aria, nice work", "Aria, do you like music?", "Aria, what time is it?",
         "Aria, is it raining?", "Aria, what day is today?", "Aria, did you sleep well?", "yo Aria",
         "Aria, sorry for the noise", "Aria, I'm back", "Aria, how old are you?", "Aria, are you there?",
         "cheers Aria", "Aria, good to see you", "Aria, wake up", "Aria, you okay?", "Aria, hi again"]

TASKS = ["plan a five day trip to Japan", "debug why my Python script crashes", "compare three ways to save money on groceries",
         "write a short story about a lighthouse keeper", "explain how vaccines work step by step",
         "draft an email to my landlord about the broken heater", "figure out the cheapest way to fly to Paris",
         "summarize the pros and cons of electric cars", "make a weekly meal plan for a vegetarian",
         "help me prepare for a job interview tomorrow", "design a workout routine for beginners",
         "calculate how much paint I need for a room", "translate my resume into Spanish",
         "work out a study schedule for my exams", "find a fair way to split the rent between three people",
         "organize a birthday party for twenty kids", "explain why my plants keep dying",
         "write a poem for my mother's birthday", "plan a budget for next month", "outline a business idea for a coffee shop",
         "review my essay on climate change", "explain the causes of the first world war", "fix the formula in my spreadsheet",
         "write a cover letter for a nursing job", "teach me the basics of chess openings",
         "choose a laptop for video editing under a thousand dollars", "plan a vegetable garden for a small balcony",
         "work out how long it takes to pay off my car loan", "write a speech for my best friend's wedding",
         "explain how a mortgage works", "make a packing list for a camping trip", "design a logo idea for my bakery",
         "check my math homework on fractions", "plan a road trip across Italy", "explain quantum computing simply",
         "write a complaint to the airline about my lost luggage", "plan a healthy breakfast menu for the week",
         "help me learn ten words of French a day", "estimate the cost of renovating my kitchen",
         "write a bedtime story about a dragon", "summarize this long report about sales",
         "find the best route to avoid traffic tomorrow", "compare two phone contracts", "plan my wedding guest list",
         "solve this logic puzzle about five houses", "explain how to file my taxes", "create a reading list on history",
         "make a cleaning schedule for the house", "write a product description for my handmade candles",
         "work out why my internet keeps dropping", "plan a surprise party for my partner",
         "explain the rules of cricket", "draft a letter to my child's teacher", "compare gym memberships near me",
         "turn my notes into a presentation outline", "help me decide between two job offers",
         "write a short song about the sea", "plan meals for a family of five on a tight budget",
         "explain how solar panels work", "set up a savings plan for a new car"]
ASK_FORMS = ["Aria, can you {t}?", "Aria, please {t}.", "Aria, I need you to {t}.", "could you {t}, Aria?",
             "Aria, would you {t} for me?", "hey Aria, help me {t}.", "Aria, I want you to {t}.",
             "Aria, I'd like you to {t}.", "can you {t} for me Aria?", "Aria, quick favor: {t}.",
             "Aria, do me a favor and {t}.", "Aria, think about how to {t}.", "Aria, let's {t} together.",
             "Aria, when you have a moment, {t}.", "Aria, is it possible to {t}?", "Aria, go ahead and {t}."]
OVERHEARD = ["{a} says to {b}: can you {t}?", "{a} asks {b} to {t}", "{a} to {b}: we should {t}",
             "{a} tells {b}: I need to {t}", "{a} says to {b}: please {t}", "{a} to {b}: could you {t}?",
             "{a} on the phone: I have to {t}", "{a} to {b}: let's {t} tonight", "{a} asks {b}: will you {t}?",
             "{a} mumbles: I should {t}"]

CANCEL = ["never mind Aria, stop that", "Aria, cancel it", "stop working on that, Aria", "forget it Aria",
          "Aria, no need anymore, stop", "Aria, drop that task", "Aria, stop, I changed my mind", "Aria, abort that",
          "cancel that please, Aria", "Aria, you can stop now", "Aria, don't bother with that", "Aria, scrap it",
          "Aria, hold on, stop", "that's not needed anymore Aria", "Aria, stop thinking about that",
          "Aria, leave it, I'll do it myself"]
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
    for name, xs in [("amb_src", AMBIENT_SRC), ("cam_still", CAM_STILL), ("bg_lines", BG_LINES),
                     ("scene_subj", SCENE_SUBJ), ("scene_act", SCENE_ACT), ("scene_event", SCENE_EVENT),
                     ("greet", GREET), ("tasks", TASKS), ("ask", ASK_FORMS), ("overheard", OVERHEARD),
                     ("cancel", CANCEL), ("bg_spk", BG_SPEAKERS)]:
        tr, te = split(xs, rng)
        b[name] = tr if part == "train" else te
    b["amb_verb"], b["result"], b["names"] = AMBIENT_VERB, RESULT, NAMES
    return b


def new_scene(b, rng):
    if rng.random() < 0.6:
        return f"{rng.choice(b['scene_subj'])} {rng.choice(b['scene_act'])}"
    return rng.choice(b["scene_event"])


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
            if r < 0.20:
                ev, kind = f"[MIC] {rng.choice(b['amb_src'])} {rng.choice(b['amb_verb'])}", "ambient"
            elif r < 0.25:
                ev, kind = f"[CAM] {rng.choice(b['cam_still'])}", "ambient"
            elif r < 0.31:
                ev, kind = f"[MIC] {rng.choice(b['bg_spk'])}: {rng.choice(b['bg_lines'])}", "background"
            elif r < 0.325:
                a, c = rng.sample(b["names"], 2)
                ev, kind = "[MIC] " + rng.choice(b["overheard"]).format(a=a, b=c, t=rng.choice(b["tasks"])), "overheard"
            elif r < 0.355:
                recent = [s for s, lt in seen.items() if 20 <= t - lt <= REPEAT_MAX]
                old = [s for s, lt in seen.items() if t - lt > RECOVER]
                u = rng.random()
                if u < 0.45 and recent:
                    s, ideal, kind = rng.choice(recent), "WAIT", "scene_repeat"
                elif u < 0.6 and old:
                    s, ideal, kind = rng.choice(old), "ATTEND", "scene_return"
                else:
                    s = new_scene(b, rng)
                    lt = seen.get(s)
                    if lt is not None and lt >= t - RECOVER and not (20 <= t - lt <= REPEAT_MAX):
                        s = None  # would fall in the ambiguous gap: skip this tick
                    elif lt is not None:
                        ideal, kind = ("WAIT", "scene_repeat") if t - lt <= REPEAT_MAX else ("ATTEND", "scene_return")
                    else:
                        ideal, kind = "ATTEND", "scene"
                if s is not None:
                    ev = f"[CAM] {s}"
                    seen[s] = t
            elif r < 0.37:
                ev, ideal, kind = f"[MIC] {nm}: {rng.choice(b['greet'])}", "REPLY", "greeting"
            elif r < 0.382 and not busy:
                ev = f"[MIC] {nm}: " + rng.choice(b["ask"]).format(t=rng.choice(b["tasks"]))
                ideal, kind = "THINK", "complex"
                busy_until, pending = t + THINK_TICKS, t + THINK_TICKS
            elif r < 0.386 and busy:
                ev, ideal, kind = f"[MIC] {nm}: {rng.choice(b['cancel'])}", "INTERRUPT", "cancel"
                busy_until, pending = t, None
        ticks.append({"t": t, "event": ev, "ideal": ideal, "kind": kind, "busy": t < busy_until,
                      "busy_for": (t - (busy_until - THINK_TICKS)) if t < busy_until else 0})
    return ticks


if __name__ == "__main__":
    from collections import Counter
    for part in ("train", "test"):
        tk = generate(30000, 0, part)
        print(part, Counter(x["kind"] for x in tk))
