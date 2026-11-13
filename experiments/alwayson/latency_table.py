"""Reading time per answer for the 21-day always-on runs E44-E49 (added 2026-11-13 for the open-source write-up).

Median, 90th percentile and maximum of mem_calls.read_s, and median index query and write time.
    python latency_table.py > latency_table.txt
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
for tag in ["e44_read", "e45_read_idle", "e46_index", "e47_nogate", "e48_full", "e49_final"]:
    d = json.load(open(os.path.join(HERE, "results", f"run_Dd_21d_s0_{tag}.json"), encoding="utf-8"))
    rs = sorted(d["mem_calls"]["read_s"])
    p90 = rs[int(0.9 * (len(rs) - 1))]
    iq, iw = d["mem_calls"].get("index_q_s") or [], d["mem_calls"].get("index_w_s") or []
    ms = lambda xs: f"{1000 * st.median(xs):.0f} ms" if xs else "-"
    print(f"{tag}: reads {len(rs)}, median {st.median(rs):.2f} s, p90 {p90:.2f} s, max {rs[-1]:.2f} s, "
          f"index query {ms(iq)}, index write {ms(iw)}")
