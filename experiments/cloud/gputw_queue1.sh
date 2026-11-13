#!/bin/bash
# GPUtw batch 1 (realistic placements, 1200 events, D=65536). Weakest-token confidence is logged on every final recall
# (FMEM_CONF_PMIN=1e-9 never rejects), so thresholds can be studied afterwards.
cd /workspace
B="SLEEP_BIG=1 SLEEP_REAL=1 FMEM_ANCHOR_OFFSET=0 FMEM_ERR_DRIVEN=1 FMEM_KEY=cross FMEM_SHIFT_NOTOK=1 FMEM_CONF_PMIN=1e-9"
C="python3 sleep.py --core 60 --noise 1140"
./gputw_run.sh real_sleep      "$C --mode sleep" $B SLEEP_TAG=_real
./gputw_run.sh real_sleep_echo "$C --mode sleep" $B FMEM_ECHO=1 SLEEP_TAG=_real_echo
./gputw_run.sh real_never      "$C --mode never" $B SLEEP_TAG=_real_never
./gputw_run.sh real_never_echo "$C --mode never" $B FMEM_ECHO=1 SLEEP_TAG=_real_never_echo
echo QUEUE1 DONE
