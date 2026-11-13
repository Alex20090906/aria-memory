#!/bin/bash
# run from the session root
P=$(pwd)/experiments/venv/Scripts/python.exe
O=experiments/memory/results_inllm
for s in 0 1 2; do
  for kind in sparse lora; do
    $P experiments/memory/inllm.py --kind $kind --seed $s --code_from mean --code_layer 14 --out $O 2>&1 | grep -E "event 0|Error" 
  done
  $P experiments/memory/inllm.py --kind sparse --seed $s --code_from mean --code_layer 14 --center 0 --tag _nocenter --out $O 2>&1 | grep -E "Error"
done
echo done
