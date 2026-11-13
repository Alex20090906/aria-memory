"""Sleep consolidation on the LLM server fast-weight memory (Gemini round 6 design, H6/H7).

Per unit j of the D anchor units: theta_j (familiarity, written by fmem_anchor.write), tau_j (decay time constant,
fast 40 or slow 500), c_j (times retrieved while awake).  A ring buffer holds the two entity names per event (owner, obj)
and nothing else (no answer text, no vector).

Wake, per write event: B[:, j] *= exp(-1/tau_j) for every unit; delta-rule write (error-driven); theta mark;
push (owner, obj) tokens; a successful gated recall bumps c on that fact's anchor units.
Sleep, every T_SLEEP events: pick replay set R* (mean theta >= 0.5 and >= half the anchor units retrieved at least
once); generative replay: the model itself completes "[owner|obj] ->" (value is NOT stored anywhere); the units the
replay walks over form U_cons; all other units are downscaled x0.1; the replayed facts are re-written with the
error-driven delta rule (only the decayed difference is refilled) and their units promoted to tau_slow.

Test (discriminates consolidation from "never decaying"): 200 events = 20 core facts (each asked once within 10
events of its write) + 180 one-off noise facts; sleep every 50; at the end recall the 20 core facts.
Conditions: fast (tau 40, no sleep), never (tau inf, no sleep), sleep (tau 40 -> 500 + downscale).
"""
import argparse
import json
import math
import os
import random

import numpy as np

import fmem_anchor as fa
import fmem_tool as f
from exp_c import NAMES as N0, OBJS as O0, PLACES, QFORMS
# larger entity pool so 500 facts fit (24 x 24 = 576 pairs)
NAMES = N0 + ["Ben", "Chloe", "Dan", "Emma", "Finn", "Grace", "Hugo", "Iris", "Jack", "Kate"]
OBJS = O0 + ["keys", "hat", "tablet", "lighter", "pen", "bike lock", "sunglasses", "flask"]
if os.environ.get("SLEEP_BIG") == "1":     # long-horizon runs: 60 x 48 = 2880 pairs
    NAMES = NAMES + ["Liam", "Noah", "Olivia", "Ava", "Lucas", "Mason", "Ella", "Zoe", "Ryan", "Nora", "Oscar", "Ruby",
                     "Sam", "Tara", "Uma", "Vera", "Will", "Xena", "Yara", "Zack", "Adam", "Beth", "Carl", "Dora",
                     "Eli", "Fay", "Gus", "Hal", "Ines", "Joel", "Kai", "Lily", "Max", "Nico", "Owen", "Pia"]
    OBJS = OBJS + ["backpack", "phone", "remote", "mug", "book", "jacket", "scissors", "tape", "flashlight", "batteries",
                   "candle", "wallet chain", "earrings", "bracelet", "helmet", "gloves", "map", "coin", "stamp", "comb",
                   "lipstick", "perfume", "toothbrush", "razor"]
PLACES = PLACES + ["in the sock drawer", "in the cookie tin", "under the bed", "behind the curtain", "on the bookshelf",
                   "in the car trunk", "in the shoe box", "inside the piano", "under the sink", "in the coat pocket",
                   "in the fruit bowl", "on the balcony", "in the attic box", "in the bread bin", "under the rug",
                   "in the guitar case"]

# U17: realistic placements. Every object only goes to places where people really keep it, with several plausible
# options each, so the model's prior cannot simply guess the answer (measured by the no-memory baseline below).
REAL = os.environ.get("SLEEP_REAL") == "1"
REAL_PLACES = {
    "passport": ["in the desk drawer", "in the bedroom safe", "in the document folder", "in the suitcase pocket", "in the filing cabinet"],
    "spare key": ["under the doormat", "in the kitchen drawer", "on the key hook", "in the flower pot", "at the neighbor's house"],
    "watch": ["on the nightstand", "in the jewelry box", "on the bathroom shelf", "in the desk drawer", "on the dresser"],
    "glasses": ["on the nightstand", "in the glasses case", "on the desk", "on the bathroom counter", "next to the sofa"],
    "ticket": ["in the wallet", "in the desk drawer", "on the fridge door", "in the backpack", "in the coat pocket"],
    "charger": ["in the desk drawer", "next to the bed", "in the backpack", "on the kitchen counter", "in the living room outlet"],
    "ring": ["in the jewelry box", "on the bathroom shelf", "on the nightstand", "in the dresser drawer", "in the soap dish"],
    "wallet": ["on the entryway table", "in the coat pocket", "in the backpack", "on the desk", "in the car"],
    "umbrella": ["in the umbrella stand", "next to the front door", "in the car", "in the backpack", "on the balcony"],
    "headphones": ["in the backpack", "on the desk", "next to the bed", "in the desk drawer", "on the sofa"],
    "notebook": ["on the desk", "in the backpack", "on the bookshelf", "in the desk drawer", "on the nightstand"],
    "camera": ["in the camera bag", "on the bookshelf", "in the closet", "in the desk drawer", "in the backpack"],
    "scarf": ["in the closet", "on the coat rack", "in the dresser drawer", "in the coat pocket", "on the chair"],
    "medicine": ["in the medicine cabinet", "in the kitchen drawer", "on the nightstand", "in the fridge", "in the first aid box"],
    "bank card": ["in the wallet", "in the desk drawer", "in the phone case", "in the document folder", "in the coat pocket"],
    "laptop": ["on the desk", "in the backpack", "on the sofa", "on the dining table", "in the laptop bag"],
    "keys": ["on the key hook", "on the entryway table", "in the coat pocket", "in the backpack", "on the kitchen counter"],
    "hat": ["on the coat rack", "in the closet", "on the shelf by the door", "in the car", "in the dresser drawer"],
    "tablet": ["on the sofa", "on the nightstand", "on the desk", "in the backpack", "on the dining table"],
    "lighter": ["in the kitchen drawer", "on the balcony", "in the coat pocket", "next to the stove", "in the car"],
    "pen": ["on the desk", "in the pencil cup", "in the desk drawer", "in the backpack", "next to the phone"],
    "bike lock": ["on the bike", "in the garage", "in the backpack", "next to the front door", "on the balcony"],
    "sunglasses": ["in the car", "on the entryway table", "in the glasses case", "in the backpack", "on the dresser"],
    "flask": ["in the kitchen cabinet", "in the backpack", "on the drying rack", "in the car", "on the desk"],
    "backpack": ["in the closet", "next to the front door", "on the desk chair", "in the car", "on the bedroom floor"],
    "phone": ["on the nightstand", "on the desk", "on the sofa", "in the coat pocket", "on the kitchen counter"],
    "remote": ["on the sofa", "on the coffee table", "next to the TV", "on the armchair", "in the TV cabinet drawer"],
    "mug": ["in the kitchen cabinet", "on the desk", "in the sink", "on the drying rack", "on the coffee table"],
    "book": ["on the bookshelf", "on the nightstand", "on the desk", "in the backpack", "on the sofa"],
    "jacket": ["in the closet", "on the coat rack", "on the chair", "in the car", "behind the door"],
    "scissors": ["in the kitchen drawer", "in the desk drawer", "in the pencil cup", "in the sewing box", "in the toolbox"],
    "tape": ["in the desk drawer", "in the toolbox", "in the kitchen drawer", "in the storage box", "on the shelf"],
    "flashlight": ["in the kitchen drawer", "in the toolbox", "next to the bed", "in the car", "in the storage cabinet"],
    "batteries": ["in the kitchen drawer", "in the toolbox", "in the desk drawer", "in the storage box", "in the TV cabinet drawer"],
    "candle": ["on the dining table", "in the kitchen drawer", "on the bathroom shelf", "in the storage box", "on the windowsill"],
    "wallet chain": ["in the dresser drawer", "on the dresser", "in the jewelry box", "in the jeans pocket", "on the desk"],
    "earrings": ["in the jewelry box", "on the dresser", "on the bathroom shelf", "on the nightstand", "in the makeup bag"],
    "bracelet": ["in the jewelry box", "on the nightstand", "on the dresser", "on the bathroom shelf", "in the makeup bag"],
    "helmet": ["in the garage", "on the bike", "next to the front door", "in the closet", "on the scooter"],
    "gloves": ["in the coat pocket", "in the closet", "on the shelf by the door", "in the car", "in the dresser drawer"],
    "map": ["in the car", "in the desk drawer", "in the backpack", "on the bookshelf", "in the document folder"],
    "coin": ["in the coin jar", "in the wallet", "on the dresser", "in the coat pocket", "in the car"],
    "stamp": ["in the desk drawer", "in the document folder", "in the wallet", "in the stationery box", "on the desk"],
    "comb": ["on the bathroom counter", "in the bathroom drawer", "on the dresser", "in the makeup bag", "in the backpack"],
    "lipstick": ["in the makeup bag", "on the bathroom counter", "on the dresser", "in the handbag", "in the coat pocket"],
    "perfume": ["on the dresser", "on the bathroom shelf", "in the makeup bag", "in the handbag", "on the nightstand"],
    "toothbrush": ["in the bathroom cup", "on the bathroom counter", "in the travel bag", "in the bathroom cabinet", "next to the sink"],
    "razor": ["in the bathroom cabinet", "on the bathroom counter", "in the shower", "in the travel bag", "in the bathroom drawer"],
}

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_FAST, TAU_SLOW, DOWNSCALE = 100.0, 500.0, 0.1
T_SLEEP = int(os.environ.get("SLEEP_EVERY", "50"))         # events between sleeps
# soft protection of consolidated rows: after a new write, a promoted row keeps (1-alpha) of its old value and takes
# only alpha of the change. alpha 0 = the hard freeze that was refuted; 1 = no protection.
PROTECT_ALPHA = float(os.environ.get("SLEEP_PROTECT_ALPHA", "0"))
# U15 in the long-term test: the network keeps only the label and its familiarity; the exact content comes from an
# append-only event log opened by that label. Recall = familiarity gate passes -> latest logged value for the label.
LOG = os.environ.get("SLEEP_LOG") == "1"
PROTECT = os.environ.get("SLEEP_PROTECT", "0") == "1"
DECAY_EVERY = int(os.environ.get("SLEEP_DECAY_EVERY", "1"))
THETA_DOWN = os.environ.get("SLEEP_THETA_DOWN", "0") == "1"   # downscale familiarity with the weights (U16)
import re
PLACE_RE = re.compile(r"^(in|on|under|behind|inside|at|next to|near|beside|on top of) [A-Za-z][\w ,'-]{2,40}$", re.I)


class Sleeper:
    def __init__(self, mode):
        self.mode = mode
        self.n_mem, self.D, self.k, self.n, _ = f.read_header()
        lay, _ = f.mem_layout(self.n_mem, self.D, self.n)
        self.boff = lay[0][3]
        self.E, self.P, self.st = fa.setup(self.D, self.k)
        self.tau = np.full(self.D, TAU_FAST if mode != "never" else 1e12, np.float32)
        self.c = np.zeros(self.D, np.int32)
        self.ring = []                      # (owner_tok_ids, obj_tok_ids) as python tuples of ints
        self.truth = {}                     # logging only: (owner, obj) -> place for core facts
        self.log = {}                       # U15: append-only event log, read back by label (latest value)
        self.events = 0
        self.vb = f.id_to_bytes()

    def B(self):
        mm = np.memmap(f.MEM, dtype=np.uint8, mode="r+")
        return mm, np.ndarray((self.D, self.n), np.float16, mm, self.boff)

    def decay(self):
        # batched decay: one pass every DECAY_EVERY events with the summed exponent (exact for pure decay; a fact
        # written in between decays at most DECAY_EVERY-1 steps less, small next to tau 100)
        self._pend = getattr(self, "_pend", 0) + 1
        if self._pend < DECAY_EVERY:
            return
        steps, self._pend = self._pend, 0
        if f.ROWSCALE:
            # lazy decay: only the D row scales change, the stored rows stay (the engine multiplies at read)
            rs = f.rs_load(self.D)
            rs *= np.exp(-float(steps) / self.tau).astype(np.float32)
            f.rs_save(rs)
            return
        mm, B = self.B()
        fac = np.exp(-float(steps) / self.tau).astype(np.float16)[:, None]
        B[:] = (B.astype(np.float32) * fac).astype(np.float16)
        mm.flush(); del B, mm
        f.touch_mem()

    def units_of(self, owner, obj):
        toks, *_ = f.capture_request(f"[{owner}|{obj}] ->")
        return fa.anchor_units(self.E, self.P, self.st, toks, owner, obj, self.k), toks

    def write(self, owner, obj, place):
        if self.mode != "never":
            self.decay()
        prot = None
        if PROTECT and self.mode == "sleep":
            # null-space protection of consolidated memory: rows of promoted (slow) units are frozen against new
            # writes, so later events cannot slowly overwrite an old consolidated fact
            prot = np.flatnonzero(self.tau >= TAU_SLOW)
            if prot.size:
                mm, B = self.B(); saved = B[prot].copy(); del B, mm
                saved_rs = f.rs_load(self.D)[prot]
        n_w = fa.write(owner, obj, place)
        if prot is not None and prot.size:
            mm, B = self.B()
            rs = f.rs_load(self.D)
            if PROTECT_ALPHA > 0:
                # blend in effective (row x row-scale) space, then store with scale 1
                old_eff = saved.astype(np.float32) * saved_rs[:, None]
                new_eff = B[prot].astype(np.float32) * rs[prot][:, None]
                B[prot] = (old_eff + PROTECT_ALPHA * (new_eff - old_eff)).astype(np.float16)
                rs[prot] = 1.0
            else:
                B[prot] = saved
                rs[prot] = saved_rs
            mm.flush(); del B, mm
            f.rs_save(rs)
            f.touch_mem(prot)
        anc, toks = self.units_of(owner, obj)
        self.log[(owner, obj)] = place          # U15 event log (latest entry per label)
        # the index keeps the two entity names (a few bytes); token ids alone lose the first letter when "[" and the
        # name's first letter merge into one token ("[Y" + "uki")
        self.ring.append((owner, obj))
        self.ring = self.ring[-64:]
        self.events += 1
        if self.mode == "sleep" and self.events % T_SLEEP == 0:
            self.sleep()
        return n_w

    def recall(self, owner, obj, question=None, bump=True):
        anc, _ = self.units_of(owner, obj)
        fam = fa.familiarity(anc, self.D)
        if fam < 0.5:
            return "", fam
        if LOG:
            if bump:
                self.c[anc] += 1
            return self.log.get((owner, obj), ""), fam
        prompt = f.PRE + ((question + "\n") if question else "") + f"[{owner}|{obj}] ->"
        out = fa.recall(prompt, owner, obj, max_tokens=8).strip()
        if bump:
            self.c[anc] += 1
        return out, fam

    def text(self, ids):
        return b"".join(self.vb.get(i, b"") for i in ids).decode("utf-8", "replace").strip()

    def sleep(self):
        th = fa.theta_load(self.D)
        replay = []
        for owner, obj in self.ring:
            anc, _ = self.units_of(owner, obj)
            if th[anc].mean() >= 0.5 and (self.c[anc] >= 1).mean() >= 0.5:
                replay.append((owner, obj, anc))
        # generative replay: the model completes each anchor from the (decayed) memory
        cons = set(); rewrites = []
        for owner, obj, anc in replay:
            out, _ = self.recall(owner, obj, bump=False)
            val = out.split("\n")[0].rstrip(".").strip()
            # reality monitoring: a replay that does not look like a place is a failed retrieval, not a memory;
            # consolidating it would turn noise into a permanent false memory
            if not PLACE_RE.match(val):
                print(f"    replay {owner} {obj}: {val!r} rejected", flush=True)
                continue
            rewrites.append((owner, obj, val))
            tr = self.truth.get((owner, obj))
            print(f"    replay {owner} {obj}: {val!r}" + (f" truth={tr!r} ok={val.lower().startswith(tr.lower())}" if tr else " (noise)"), flush=True)
            cons.update(anc.tolist())
            # answer rows: shifted units for each token of the completion (same hash as the engine)
            toks, *_ = f.capture_request(f"[{owner}|{obj}] -> {val}.")
            pos = fa.token_ids_of(toks, f"[{owner}|{obj}] -> {val}.")
            a_end = max(i for i in pos if b"->" in self.vb.get(int(toks[i]), b""))
            for t in range(a_end + 1, pos[-1]):
                cons.update(fa.shifted(anc, t - a_end, toks[t], self.D).tolist())
        # synaptic downscaling of everything not consolidated
        mm, B = self.B()
        # units already promoted to tau_slow are "cortical": never downscaled again (a consolidated fact leaves the
        # ring after 64 events, so it would otherwise be erased at the next sleep)
        mask = self.tau < TAU_SLOW
        mask[list(cons)] = False
        # events of the last wake window are still "unjudged" (not yet asked): keep their rows intact but promote
        # nothing. Their answer rows depend on the stored tokens, which we do not keep, so they are re-derived from a
        # gated recall of each fact (the hippocampal index re-cues the cortex).
        for owner, obj in self.ring[-T_SLEEP:]:
            anc, _ = self.units_of(owner, obj)
            mask[anc] = False
            out, _ = self.recall(owner, obj, bump=False)
            val = out.split("\n")[0].rstrip(".").strip()
            if val:
                text = f"[{owner}|{obj}] -> {val}."
                t2, *_ = f.capture_request(text)
                pos = fa.token_ids_of(t2, text)
                a_end = max(i for i in pos if b"->" in self.vb.get(int(t2[i]), b""))
                for t in range(a_end + 1, pos[-1]):
                    mask[fa.shifted(anc, t - a_end, t2[t], self.D)] = False
        if f.ROWSCALE:
            del B, mm
            rs = f.rs_load(self.D); rs[mask] *= np.float32(DOWNSCALE); f.rs_save(rs)
        else:
            B[mask] = (B[mask].astype(np.float32) * DOWNSCALE).astype(np.float16)
            mm.flush(); del B, mm
            f.touch_mem()
        if THETA_DOWN:
            # U16 theorem 0.1: familiarity must fade together with the content it stands for; downscaled units lose
            # their familiarity by the same factor, so a faded trace no longer opens the gate
            th = fa.theta_load(self.D)
            th[mask] *= np.float32(DOWNSCALE)
            np.save(fa.THETA, th)
        # refill (error-driven delta rule only tops up the decayed difference) and promote
        for owner, obj, val in rewrites:
            fa.write(owner, obj, val)
        if cons:
            self.tau[list(cons)] = TAU_SLOW
        np.save(fa.TAU_FILE, self.tau)   # fmem_anchor slows theta decay on promoted units
        print(f"  sleep @ {self.events}: replayed {len(rewrites)}/{len(replay)} of {len(self.ring)} ring, "
              f"consolidated units {len(cons)}", flush=True)


def main(a):
    rng = random.Random(a.seed)
    pairs = [(n, o) for n in NAMES for o in OBJS]; rng.shuffle(pairs)
    place = (lambda o: rng.choice(REAL_PLACES[o])) if REAL else (lambda o: rng.choice(PLACES))
    core = [(n, o, place(o)) for n, o in pairs[:a.core]]
    noise = [(n, o, place(o)) for n, o in pairs[a.core:a.core + a.noise]]
    # schedule: core facts spread evenly, each asked once 1..10 events later
    total = a.core + a.noise
    sched = [None] * total
    core_pos = sorted(rng.sample(range(total), a.core))
    for i, p in enumerate(core_pos):
        sched[p] = ("core", core[i])
    ni = iter(noise)
    for t in range(total):
        if sched[t] is None:
            sched[t] = ("noise", next(ni))
    asks = {}
    for i, p in enumerate(core_pos):
        asks.setdefault(min(total - 1, p + rng.randint(1, 10)), []).append(core[i])
    f.reset(None); fa.theta_reset(); fa.off()
    S = Sleeper(a.mode)
    S.truth = {(n, o): p for n, o, p in core}
    for t, (kind, (n, o, p)) in enumerate(sched):
        S.write(n, o, p)
        for (cn, co, cp) in asks.get(t, []):
            out, fam = S.recall(cn, co, rng.choice(QFORMS).format(n=cn, o=co))
            print(f"  ask@{t} {cn} {co}: {out[:30]!r} ok={out.lower().startswith(cp.lower())}", flush=True)
        if t % 20 == 0:
            print(f"event {t}/{total} ({kind})", flush=True)
    def fam_both(n, o):
        """both familiarity readings for the 2x2 ablation (Gemini round 10): theta mean and key-echo"""
        anc, _ = S.units_of(n, o)
        th = float(fa.theta_load(S.D)[anc].mean())
        ec = float(fa.echo_familiarity(anc, S.D)) if fa.ECHO else None
        return round(th, 3), (round(ec, 3) if ec is not None else None)

    res = []
    for i, (n, o, p) in enumerate(core):
        fa.LAST_PMIN = None
        out, fam = S.recall(n, o, rng.choice(QFORMS).format(n=n, o=o), bump=False)
        ok = out.lower().startswith(p.lower())     # places like "next to the TV" have capitals
        th, ec = fam_both(n, o)
        res.append({"fact": f"{n}'s {o}: {p}", "out": out[:40], "theta": round(fam, 3), "ok": ok, "pmin": fa.LAST_PMIN,
                    "theta_mean": th, "echo": ec,
                    "age": total - core_pos[i], "honest_forget": (not ok) and out.strip() == "",
                    "confident_wrong": (not ok) and out.strip() != ""})
    # never-told probes: the same kind of question about owner/object pairs that were never written. Anything other
    # than an empty answer here is a false memory (familiarity gate or content confidence failed).
    never = []
    for n, o in pairs[a.core + a.noise:a.core + a.noise + a.core]:
        fa.LAST_PMIN = None
        out, fam = S.recall(n, o, rng.choice(QFORMS).format(n=n, o=o), bump=False)
        th, ec = fam_both(n, o)
        never.append({"pair": f"{n}|{o}", "out": out[:40], "gate": round(fam, 3), "pmin": fa.LAST_PMIN,
                      "theta_mean": th, "echo": ec, "answered": out.strip() != ""})
    print(f"  never-told answered (false memories): {sum(r['answered'] for r in never)}/{len(never)}", flush=True)
    # no-memory baseline: same questions with the memory switched off. With realistic placements the model may guess
    # right from common sense alone; that share must not be credited to memory.
    fa.off()
    for (n, o, p), r in zip(core, res):
        q = rng.choice(QFORMS).format(n=n, o=o)
        g = f.chat([{"role": "user", "content": f.RAW + f.PRE + q + "\n" + f"[{n}|{o}] ->"}], max_tokens=8).strip()
        r["prior_guess"] = g[:40]
        r["prior_ok"] = g.lower().startswith(p.lower())
    print(f"  no-memory guess baseline: {sum(r['prior_ok'] for r in res)}/{len(res)}", flush=True)
    acc = float(np.mean([r["ok"] for r in res]))
    print(f"MODE {a.mode}: core recall at end {acc:.3f} ({sum(r['ok'] for r in res)}/{len(res)}) "
          f"honest-forget {sum(r['honest_forget'] for r in res)} confident-wrong {sum(r['confident_wrong'] for r in res)}")
    # retention by age quartile
    ages = sorted(res, key=lambda r: r["age"])
    qn = max(1, len(ages) // 4)
    print("  recall by age quartile (young->old):",
          [f"{sum(r['ok'] for r in ages[j * qn:(j + 1) * qn])}/{len(ages[j * qn:(j + 1) * qn])}" for j in range(4)])
    json.dump({"mode": a.mode, "core": a.core, "noise": a.noise, "acc": acc, "rows": res, "never": never},
              open(os.path.join(HERE, f"sleep_{a.mode}_n{a.noise}{os.environ.get('SLEEP_TAG', '')}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "never", "sleep"], default="sleep")
    ap.add_argument("--core", type=int, default=20)
    ap.add_argument("--noise", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
