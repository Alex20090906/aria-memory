"""Step A on the AI PC (research server on 8093 already loaded, memory.bin v2 at layer 38):
calibrate mu/sd -> 8-fact test (forward and reversed order) without BCM -> iterative BCM -> same tests.
Writes exp_a.json next to this file."""
import argparse
import json
import os

import fmem_tool as f

HERE = os.path.dirname(os.path.abspath(__file__))
SENT = os.path.join(HERE, "calib_sentences.txt")
res = {}


def run_test(tag, reverse, dl):
    a = argparse.Namespace(gain=0.0, calibrate=0, head=1, dl=dl, reverse=reverse, tag=tag)
    f.test(a)
    out = json.load(open(os.path.join(HERE, f"test_head1_dl{dl}_rev{reverse}_{tag}.json"), encoding="utf-8"))
    res[f"{tag}_rev{reverse}_dl{dl}"] = {"cued": out["recall_cued"], "qa": out["recall"],
                                         "generic": out["generic_after"],
                                         "answers": [q["cued"].strip()[:40] for q in out["questions"]]}
    print(tag, reverse, dl, out["recall_cued"], flush=True)


f.calibrate(argparse.Namespace(sentences=SENT))
for rev in (0, 1):
    run_test("nobcm", rev, 14.0)
f.bcm(argparse.Namespace(sentences=SENT, epochs=5, eta=0.3, clip=3.0))
for rev in (0, 1):
    run_test("bcm", rev, 14.0)
run_test("bcm", 0, 10.0)
json.dump(res, open(os.path.join(HERE, "exp_a.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
