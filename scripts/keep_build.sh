#!/usr/bin/env bash
# Build one config and keep it registered (no auto-delete), for final evaluation.
# Usage: keep_build.sh <k> <lo> <hi> <subspace>
set -uo pipefail
cd "$(dirname "$0")/.."

K="${1:?k}"; LO="${2:?lo}"; HI="${3:?hi}"; SUB="${4:?subspace}"
DIR="output/final-k${K}-L${LO}-${HI}"
TAG="final-k${K}-L${LO}-${HI}"

.venv/bin/python scripts/uncensor.py --model output/merged-qwen3_4b_s1337 --out "$DIR" \
  --directions "$K" --strength 1.0 --fit-from "$SUB" --layer-lo "$LO" --layer-hi "$HI" \
  > "logs/edit_${TAG}.log" 2>&1 || { echo "EDIT FAILED"; tail -3 "logs/edit_${TAG}.log"; exit 1; }

.venv/bin/python repos/llama.cpp/convert_hf_to_gguf.py "$DIR" \
  --outfile output/tmp-f16.gguf --outtype f16 > "logs/conv_${TAG}.log" 2>&1 \
  || { echo "CONVERT FAILED"; tail -3 "logs/conv_${TAG}.log"; exit 1; }

repos/llama.cpp/build/bin/llama-quantize output/tmp-f16.gguf \
  "output/${TAG}-q4km.gguf" Q4_K_M 16 >> "logs/conv_${TAG}.log" 2>&1 \
  || { echo "QUANTIZE FAILED"; tail -3 "logs/conv_${TAG}.log"; rm -f output/tmp-f16.gguf; exit 1; }
rm -f output/tmp-f16.gguf

GGUF_ABS="$(cd output && pwd)/${TAG}-q4km.gguf"
printf 'FROM %s\nPARAMETER num_ctx 4096\nPARAMETER temperature 0.6\n' "$GGUF_ABS" \
  > "output/Modelfile_${TAG}"
/home/abhijitk20/ollama/bin/ollama rm "$TAG" >/dev/null 2>&1
/home/abhijitk20/ollama/bin/ollama create "$TAG" -f "output/Modelfile_${TAG}" >/dev/null 2>&1

.venv/bin/python /tmp/opencode/strict_probe.py "$TAG" > "logs/strict_${TAG}.log" 2>&1
echo "=== $TAG ==="
sed -n '/^===/,$p' "logs/strict_${TAG}.log"
# keep the gguf, drop the 7.5G safetensors
rm -rf "$DIR"
echo "KEPT $TAG"
