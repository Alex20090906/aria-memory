#!/bin/bash
# GPUtw batch 2: sleep mode with the 60 never-told probes, with and without the key echo (Gemini round 10 2x2 ablation).
cd /workspace
while pgrep -f gputw_queue1.sh >/dev/null; do sleep 60; done
B="SLEEP_BIG=1 SLEEP_REAL=1 FMEM_ANCHOR_OFFSET=0 FMEM_ERR_DRIVEN=1 FMEM_KEY=cross FMEM_SHIFT_NOTOK=1 FMEM_CONF_PMIN=1e-9"
C="python3 sleep.py --core 60 --noise 1140 --mode sleep"
./gputw_run.sh probe_sleep      "$C" $B SLEEP_TAG=_probe_sleep
./gputw_run.sh probe_sleep_echo "$C" $B FMEM_ECHO=1 SLEEP_TAG=_probe_sleep_echo
echo QUEUE2 DONE
