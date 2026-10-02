#!/usr/bin/env bash
# Score one fine-tune seed on BFCL v4 irrelevance, end to end.
#
# Why this exists: BFCL irrelevance showed v1 at 89.17% vs an untrained base at
# 87.50% -- a 4-entry gap. Before chasing a bigger number, we need to know the
# run-to-run variance. This script turns "adapter on disk" into a comparable
# score, using exactly the same conditions as the published v1 run.
#
# Usage: ./scripts/eval_seed.sh <adapter_dir> <ollama_tag> [max_tokens]
#   e.g. ./scripts/eval_seed.sh output/full4b_seed1337 qwen3-4b-s1337

set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?usage: eval_seed.sh <adapter_dir> <ollama_tag> [max_tokens]}"
TAG="${2:?usage: eval_seed.sh <adapter_dir> <ollama_tag> [max_tokens]}"
MAXTOK="${3:-512}"
NAME="${TAG}"
SLUG="${TAG//[^a-zA-Z0-9]/_}"
RESULTDIR=".venv/lib/python3.12/site-packages/result"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "!! no adapter_model.safetensors in $ADAPTER" >&2; exit 1
fi

echo "=== [1/4] merge + quantise $ADAPTER -> ${TAG}.gguf ==="
if [ ! -f "output/${TAG}.gguf" ]; then
  .venv/bin/python scripts/merge_export.py --adapter "$ADAPTER" --out "output/merged-${SLUG}" --name "${TAG}.gguf"
fi
ls -la "output/${TAG}.gguf" | awk '{printf "    %s %.2f GB\n",$9,$5/1073741824}'

echo "=== [2/4] register with ollama ==="
cat > "output/Modelfile_${SLUG}" <<EOF
FROM $(pwd)/output/${TAG}.gguf
PARAMETER num_ctx 4096
PARAMETER temperature 0.6
PARAMETER stop ""
EOF
ollama rm "$TAG" >/dev/null 2>&1 || true
ollama create "$TAG" -f "output/Modelfile_${SLUG}" >/dev/null
echo "    ollama: $TAG"

echo "=== [3/4] generate (BFCL irrelevance, 240 entries) ==="
rm -rf "${RESULTDIR:?}/${SLUG}"
# OLLAMA_* limits matter: this host has 14GB RAM and swaps out without them.
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_THREADS=2 \
OLLAMA_KEEP_ALIVE=5m OLLAMA_CONTEXT_LENGTH=4096 \
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  .venv/bin/python scripts/run_bfcl.py generate \
    --model "$SLUG" --test-category irrelevance --temperature 0

N=$(wc -l < "${RESULTDIR}/${SLUG}/non_live/BFCL_v4_irrelevance_result.json")
echo "    generated ${N}/240"
if [ "$N" -ne 240 ]; then
  echo "!! only ${N}/240 entries - BFCL refuses to score partial results. Relaunch to resume." >&2
  exit 2
fi

echo "=== [4/4] score ==="
BFCL_MAX_TOKENS="$MAXTOK" .venv/bin/python scripts/run_bfcl.py evaluate \
  --model "$SLUG" --test-category irrelevance 2>&1 | grep -E "Accuracy"