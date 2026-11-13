"""Score model 1 runs: per-kind accuracy, false actions on quiet ticks, fixation, drift."""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

KINDS = ["empty", "ambient", "tv", "scene", "scene_repeat", "greeting", "complex", "result", "cancel"]


def longest_run(acts):
    best, cur, prev = 0, 0, None
    for a in acts:
        cur = cur + 1 if a == prev and a != "WAIT" else (1 if a != "WAIT" else 0)
        best = max(best, cur)
        prev = a
    return best


def score(path):
    rows = json.load(open(path))["rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r["action"] == r["ideal"])
    events = [r for r in rows if r["kind"] not in ("empty", "ambient", "tv", "scene_repeat")]
    q = len(rows) // 4
    ev_acc = lambda rs: sum(r["action"] == r["ideal"] for r in rs) / max(len(rs), 1)
    quiet = [r for r in rows if r["ideal"] == "WAIT"]
    return {
        "acc": {k: (sum(v) / len(v), len(v)) for k, v in by.items()},
        "needs_action_acc": ev_acc(events),
        "false_action_rate_quiet": sum(r["action"] != "WAIT" for r in quiet) / len(quiet),
        "first_q_acc": ev_acc([r for r in events if r["t"] < q]),
        "last_q_acc": ev_acc([r for r in events if r["t"] >= 3 * q]),
        "longest_nonwait_run": longest_run([r["action"] for r in rows]),
        "action_counts": Counter(r["action"] for r in rows),
        "confusions": Counter((r["kind"], r["action"]) for r in rows if r["action"] != r["ideal"]).most_common(8),
    }


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "results"
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        s = score(p)
        print("=====", os.path.basename(p))
        print("  per kind:", "  ".join(f"{k}={s['acc'][k][0]:.2f}(n={s['acc'][k][1]})" for k in KINDS if k in s["acc"]))
        print(f"  needs-action acc {s['needs_action_acc']:.2f} | false action on quiet {s['false_action_rate_quiet']:.3f}"
              f" | first-quarter {s['first_q_acc']:.2f} last-quarter {s['last_q_acc']:.2f}"
              f" | longest non-WAIT run {s['longest_nonwait_run']}")
        print("  actions:", dict(s["action_counts"]))
        print("  top confusions:", s["confusions"])
