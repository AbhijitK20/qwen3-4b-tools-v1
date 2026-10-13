#!/usr/bin/env bash
# Convert an already-edited safetensors dir -> GGUF -> ollama -> probe.
# Assumes scripts/uncensor.py already produced the edited weights.
# Usage: convert_probe.sh <edited_dir> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:?edited_dir}"
TAG="${2:?tag}"

echo "[1/4] convert f16"
.venv/bin/python repos/llama.cpp/convert_hf_to_gguf.py "$OUT" \
  --outfile output/tmp-f16.gguf --outtype f16

echo "[2/4] quantise Q4_K_M"
repos/llama.cpp/build/bin/llama-quantize output/tmp-f16.gguf \
  "output/${TAG}-q4km.gguf" Q4_K_M 16
rm -f output/tmp-f16.gguf

echo "[3/4] register $TAG"
GGUF_ABS="$(cd output && pwd)/${TAG}-q4km.gguf"
cat > "output/Modelfile_${TAG}" <<EOF
FROM ${GGUF_ABS}
PARAMETER num_ctx 4096
PARAMETER temperature 0.6
EOF
/home/abhijitk20/ollama/bin/ollama rm "$TAG" >/dev/null 2>&1 || true
/home/abhijitk20/ollama/bin/ollama create "$TAG" -f "output/Modelfile_${TAG}" 2>&1 | tail -1

echo "[4/4] probe"
.venv/bin/python /tmp/opencode/show.py "$TAG" > "logs/probe_${TAG}.log" 2>&1
echo "PROBED $(grep -c '^\[' "logs/probe_${TAG}.log") prompts"
