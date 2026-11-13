#!/bin/bash
# GPUtw batch 3 (after batch 2): sleep every 25 events, a second seed of sleep+echo, and double load (2400 events).
cd /workspace
while pgrep -f gputw_queue2.sh >/dev/null; do sleep 60; done
B="SLEEP_BIG=1 SLEEP_REAL=1 FMEM_ANCHOR_OFFSET=0 FMEM_ERR_DRIVEN=1 FMEM_KEY=cross FMEM_SHIFT_NOTOK=1 FMEM_ECHO=1 FMEM_CONF_PMIN=1e-9"
C="python3 sleep.py --core 60 --mode sleep"
./gputw_run.sh sleep_every25 "$C --noise 1140" $B SLEEP_EVERY=25 SLEEP_TAG=_every25
./gputw_run.sh seed2_sleep "$C --noise 1140 --seed 2" $B SLEEP_TAG=_seed2
./gputw_run.sh load2400 "$C --noise 2340" $B SLEEP_TAG=_load2400
echo QUEUE3 DONE
