#!/bin/bash
# all runs from the session root (long-path DLL issue otherwise)
P=$(pwd)/experiments/venv/Scripts/python.exe
for enc in minilm bge qwen3emb; do
  for s in 0 1 2; do
    $P experiments/model1/train_gru2.py --encoder $enc --seed $s --out experiments/model1/results_w2 2>&1 | grep -v -i warn | tail -3
  done
done
