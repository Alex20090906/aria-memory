#!/bin/bash
# One research run on the GPUtw instance (RTX 5090): own memory dir, research server with the fast-weight memory on,
# then the experiment command. Runs are queued one after another (one GPU).
#   gputw_run.sh TAG "command" [ENV=VALUE ...]
set -u
TAG=$1; CMD=$2; shift 2
PROFILE=${PROFILE:-$(ls /workspace/llm/llm-*.json /workspace/LLM server-data/llm-*.json 2>/dev/null | head -1)}
PY=${PY:-python3}
[ -x "$PY" ] || PY=python3
RUN=/workspace/runs/$TAG
mkdir -p "$RUN" /workspace/fmem_cache
cp -n /workspace/fmem_cache/* "$RUN"/ 2>/dev/null
cd /workspace/exp
export PYTHONUTF8=1 FMEM_LLM=/workspace/llm FMEM_DATA="$RUN" FMEM_URL=http://127.0.0.1:8093/v1/chat/completions
export FMEM_GGUF=/vault/llm/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
export FMEM_TOKENIZER=$(ls /workspace/LLM server-data/packs/*/tokenizer/vocab.json 2>/dev/null | head -1)
export LLM_FMEM_FILE="$RUN/memory.bin" LLM_FMEM_CAPTURE="$RUN/capture.bin" LLM_FMEM_HEAD=1
export LLM_FMEM_ANCHOR="$RUN/anchor.bin" LLM_FMEM_DEBUG=1 LLM_FMEM_SHIFT_NOTOK=1
export LLM_FMEM_ROWSCALE="$RUN/rowscale.bin" FMEM_ROWSCALE_FILE="$RUN/rowscale.bin" FMEM_DIRTY=1
for kv in "$@"; do export "$kv"; done
echo "[$TAG] start $(date +%H:%M:%S) profile=$PROFILE py=$PY" | tee "$RUN/run.log"
fresh=0
if [ ! -f "$RUN/memory.bin" ]; then
  $PY fmem_tool.py init --layers 38 --D 65536 --k 16 >> "$RUN/run.log" 2>&1 || { echo "[$TAG] init failed"; tail -20 "$RUN/run.log"; exit 1; }
  fresh=1
fi
$PY /workspace/llm/serve/server.py --engine llm --config "$PROFILE" --port 8093 > "$RUN/serve.log" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 180); do
  sleep 10
  kill -0 $SRV 2>/dev/null || { echo "[$TAG] server exited"; tail -30 "$RUN/serve.log"; exit 1; }
  h=$(curl -s -m 5 http://127.0.0.1:8093/health || true)
  case "$h" in *'"loaded": true'*|*'"loaded":true'*) ok=1; break ;; esac
  [ $i -eq 3 ] && curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8093/v1/load >/dev/null || true
done
[ $ok = 1 ] || { echo "[$TAG] server did not load"; tail -30 "$RUN/serve.log"; kill $SRV; exit 1; }
echo "[$TAG] server loaded $(date +%H:%M:%S)" | tee -a "$RUN/run.log"
if [ $fresh = 1 ]; then
  $PY fmem_tool.py calibrate calib_sentences.txt >> "$RUN/run.log" 2>&1
fi
t0=$(date +%s)
bash -c "$CMD" > "$RUN/stdout.log" 2>&1
rc=$?
echo "[$TAG] done rc=$rc in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$RUN/run.log"
for fn in unembed_f16.npy unembed_mean.npy token_embd_f16.npy anchor_P_65536.npy anchor_stats.npz; do
  [ -f "$RUN/$fn" ] && [ ! -f "/workspace/fmem_cache/$fn" ] && cp "$RUN/$fn" /workspace/fmem_cache/
done
cp /workspace/exp/sleep_*"${SLEEP_TAG:-}".json "$RUN"/ 2>/dev/null
cp /workspace/exp/zh_result_*.json "$RUN"/ 2>/dev/null
grep -E "^MODE|quartile|baseline|^SUMMARY" "$RUN/stdout.log" | tail -6
kill $SRV; sleep 5; kill -9 $SRV 2>/dev/null
pkill -f "llm --pack" 2>/dev/null
rm -f "$RUN/memory.bin" "$RUN/capture.bin"
exit $rc
