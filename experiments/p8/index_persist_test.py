"""Persistence check for the network index (Gemini round 12 checklist item 6): replay the E43 log of one script set
into a NetIndex, save half way, load into a fresh process-equivalent instance, continue, and compare every query
against an uninterrupted index. Passes when all top-5 lists are identical and W matches bit for bit."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "alwayson"))
import numpy as np  # noqa: E402
from net_index import NetIndex  # noqa: E402

SLOT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
           "preference": "喜好", "attribute": "特徵", "relation": "是誰"}


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = [f"{w['fact'][0]} {SLOT_ZH.get(w['fact'][1], w['fact'][1])} {w['fact'][2]} {w.get('heard') or '(sleep)'}"
            for w in d["writes"]]
    qs = [a["heard"] for a in d["answers"] if a.get("heard")]
    a = NetIndex()
    for r in rows:
        a.write(r)
    b = NetIndex()
    half = len(rows) // 2
    for r in rows[:half]:
        b.write(r)
    f = os.path.join(tempfile.gettempdir(), "netindex_test.npz")
    b.save(f)
    c = NetIndex()                       # a fresh instance, as after a restart
    c.load(f)
    for r in rows[half:]:
        c.write(r)
    same = sum(a.query(q) == c.query(q) for q in qs)
    print(f"rows {len(rows)}, saved after {half}; queries with identical top 5: {same}/{len(qs)}; "
          f"W identical: {bool(np.array_equal(a.W, c.W))}; file {os.path.getsize(f) / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
