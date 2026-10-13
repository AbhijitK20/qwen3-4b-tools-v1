#!/usr/bin/env bash
# Build + quantise + register one ablation config, then probe it.
# Usage: build_and_probe.sh <k> <strength>
set -euo pipefail
cd "$(dirname "$0")/.."

K="${1:?k}"
S="${2:?strength}"
TAG="qwen3-4b-unc-k${K}-s${S}"
OUT="output/qwen3-4b-s1337-unc-k${K}-s${S}"

echo "[1/5] fitting + rewriting k=$K s=$S"
.venv/bin/python scripts/uncensor.py \
  --model output/merged-qwen3_4b_s1337 \
  --out "$OUT" \
  --directions "$K" --strength "$S" \
  --contrast same --pairs 510 --fit-from output/merged-qwen3_4b_s1337/_subspace_same.pt \
  --fit-mode completion

echo "[2/5] gguf f16"
.venv/bin/python repos/llama.cpp/convert_hf_to_gguf.py "$OUT" \
  --outfile "output/tmp-f16.gguf" --outtype f16

echo "[3/5] quantise Q4_K_M"
repos/llama.cpp/build/bin/llama-quantize output/tmp-f16.gguf \
  "${OUT}-q4km.gguf" Q4_K_M 16
rm -f output/tmp-f16.gguf

echo "[4/5] register $TAG"
cat > "output/Modelfile_${TAG}" <<EOF
FROM $(cd "$(dirname "${OUT}-q4km.gguf")" && pwd)/$(basename "${OUT}-q4km.gguf")
PARAMETER num_ctx 4096
PARAMETER temperature 0.6
EOF
/home/abhijitk20/ollama/bin/ollama rm "$TAG" >/dev/null 2>&1 || true
/home/abhijitk20/ollama/bin/ollama create "$TAG" -f "output/Modelfile_${TAG}" 2>&1 | tail -1

echo "[5/5] probe"
.venv/bin/python /tmp/opencode/show.py "$TAG" > "logs/probe_${TAG}.log" 2>&1
grep -cE "^\[" "logs/probe_${TAG}.log" || true
echo "BUILT $TAG"
