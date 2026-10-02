#!/usr/bin/env bash
# Autonomous end-to-end run: pre-flight -> resume training -> merge -> eval.
#
# Designed to run unattended with NO network. Launched via setsid+nohup so it
# survives the launching terminal closing.
#
#   setsid nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 < /dev/null &
#
# Every phase logs a banner. Phases 4-6 degrade rather than abort: a failed
# GGUF export still leaves a merged fp16 model, and checkpoints written every
# 200 steps are independently evaluable.

set -u -o pipefail

BASE="$HOME/model training"
PY="$BASE/.venv/bin/python"
TRAIN="$BASE/scripts/train_qlora_v2.py"
MERGE="$BASE/scripts/merge_export.py"
EVAL="$BASE/scripts/eval_toolcalls.py"
ADAPTER="$BASE/output/full4b"
MERGED="$BASE/output/merged-4b"

# Force offline everywhere. Any accidental fetch fails fast instead of
# hanging on a dead network for minutes per request.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL=1
export UNSLOTH_COMPILE_LOCATION="$BASE/unsloth_compiled_cache"
export PATH="$HOME/ollama/bin:$PATH"
export OLLAMA_HOST="http://127.0.0.1:11434"

banner() { echo; echo "=========== $* ==========="; date; echo; }
fail()   { echo "[run_all] FATAL: $*" | tee -a "$BASE/logs/run_all.fatal"; exit 1; }

mkdir -p "$BASE/logs" "$BASE/eval"

# ---------------------------------------------------------------- phase 3
# Offline pre-flight. This is the whole point of testing before the long run:
# if anything is genuinely missing we find out in ~3 min, not at hour five.
banner "PHASE 3  offline pre-flight (HF_HUB_OFFLINE=1)"
PRE="$BASE/logs/preflight.log"
"$PY" "$TRAIN" --out "$BASE/output/preflight" --max-steps 5 \
      --model unsloth/Qwen3-4B-unsloth-bnb-4bit \
      --seq-len 1024 --batch 4 --grad-accum 4 --lr 2e-4 \
      > "$PRE" 2>&1
RC=$?
tail -5 "$PRE"
if [ $RC -ne 0 ]; then
    if grep -qiE "offline|connection|resolve|Could not|not found|404|ENOTFOUND" "$PRE"; then
        fail "pre-flight needs network - see $PRE"
    fi
    echo "[run_all] pre-flight non-zero ($RC) but no network signature; continuing" | tee -a "$BASE/logs/run_all.warn"
fi
echo "[run_all] pre-flight ok"

# ---------------------------------------------------------------- phase 4
# Resume. Flags MUST match checkpoint-200 or the adapter is invalidated:
#   --out      default is output/qwen3-8b-tools (nonexistent) -> silent fresh run
#   --seq-len  default 2048, but checkpoint-200 trained at 1024
#   --batch/--grad-accum 4/4 -> effective batch 16, as in the checkpoint
banner "PHASE 4  resume training from checkpoint-200 -> step 1200"
"$PY" "$TRAIN" \
    --out "$ADAPTER" --resume \
    --model unsloth/Qwen3-4B-unsloth-bnb-4bit \
    --seq-len 1024 --batch 4 --grad-accum 4 \
    --max-steps 1200 --lr 2e-4 \
    > "$BASE/logs/train_resume.log" 2>&1
RC=$?
echo "[run_all] training exited rc=$RC"
grep -E "^\[5\]|train_runtime" "$BASE/logs/train_resume.log" | tail -3
if [ $RC -ne 0 ]; then
    fail "training failed rc=$RC - see $BASE/logs/train_resume.log"
fi

LATEST=$(ls -d "$ADAPTER"/checkpoint-* 2>/dev/null \
         | sed 's/.*checkpoint-//' | sort -n | tail -1)
echo "[run_all] newest checkpoint: ${LATEST:-none}"

# ---------------------------------------------------------------- phase 5
banner "PHASE 5  merge adapter -> fp16 -> GGUF -> ollama"
"$PY" "$MERGE" --adapter "$ADAPTER" --out "$MERGED" \
    > "$BASE/logs/merge.log" 2>&1
echo "[run_all] merge exited rc=$? (non-fatal: merged fp16 still on disk)"
tail -8 "$BASE/logs/merge.log"

# ---------------------------------------------------------------- phase 6
banner "PHASE 6  eval base vs fine-tuned"
pkill -f "ollama serve" 2>/dev/null
sleep 3
setsid nohup ollama serve > "$BASE/logs/ollama_serve.log" 2>&1 < /dev/null &
for i in $(seq 1 40); do
    if curl -s -m 2 http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then break; fi
    sleep 2
done
curl -s -m 5 http://127.0.0.1:11434/api/tags > "$BASE/eval/ollama_models.json" 2>&1
echo "[run_all] registered models:"
grep -o '"name":"[^"]*"' "$BASE/eval/ollama_models.json" | head

# base model reference point
"$PY" "$EVAL" --model qwen3:8b --n 120 --out "$BASE/eval/base_qwen3_8b.json" \
    > "$BASE/logs/eval_base.log" 2>&1
echo "[run_all] base eval rc=$?"

# fine-tuned model, if it registered
if grep -q "qwen3-4b-tools-q4km" "$BASE/eval/ollama_models.json" 2>/dev/null; then
    "$PY" "$EVAL" --model qwen3-4b-tools-q4km --n 120 \
        --out "$BASE/eval/finetuned_4b.json" \
        > "$BASE/logs/eval_ft.log" 2>&1
    echo "[run_all] finetuned eval rc=$?"
    tail -14 "$BASE/logs/eval_ft.log"
else
    echo "[run_all] fine-tuned model not registered; see merge.log"
fi

banner "RUN_ALL COMPLETE"
date
ls -la "$BASE/eval"
