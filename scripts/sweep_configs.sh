#!/usr/bin/env bash
# Sweep ablation configs unattended.
# Each config: edit weights (reuse cached subspace) -> GGUF -> register -> probe
# -> record verdict -> delete the safetensors + gguf so the next one has room.
#
# Usage: sweep_configs.sh "<k:s[:lo:hi]> ..."   lo/hi = layer band (default all)
set -uo pipefail
cd "$(dirname "$0")/.."

SUB="${SUB:-output/merged-qwen3_4b_s1337/_subspace_same.pt}"
BASE=output/merged-qwen3_4b_s1337
SUMMARY=logs/sweep_results.txt
: > "$SUMMARY"

for cfg in "$@"; do
  K="${cfg%%:*}"; rest="${cfg#*:}"
  S="${rest%%:*}"; band="${rest#*:}"
  if [ "$band" = "$rest" ]; then LO=0; HI=1000000000; BAND="all"
  else LO="${band%%:*}"; HI="${band##*:}"; BAND="L${LO}-${HI}"; fi
  DIR="output/unc-k${K}-s${S}-${BAND}"
  TAG="unc-k${K}-s${S}-${BAND}"

  echo "########## k=$K s=$S band=$BAND ##########"
  if [ ! -d "$DIR" ]; then
    .venv/bin/python scripts/uncensor.py --model "$BASE" --out "$DIR" \
      --directions "$K" --strength "$S" --fit-from "$SUB" \
      --layer-lo "$LO" --layer-hi "$HI" \
      > "logs/edit_${TAG}.log" 2>&1 || { echo "EDIT FAILED k=$K s=$S" | tee -a "$SUMMARY"; continue; }
  fi

  if [ ! -f "output/${TAG}-q4km.gguf" ]; then
    .venv/bin/python repos/llama.cpp/convert_hf_to_gguf.py "$DIR" \
      --outfile output/tmp-f16.gguf --outtype f16 > "logs/conv_${TAG}.log" 2>&1 \
      && repos/llama.cpp/build/bin/llama-quantize output/tmp-f16.gguf \
         "output/${TAG}-q4km.gguf" Q4_K_M 16 >> "logs/conv_${TAG}.log" 2>&1 \
      || { echo "CONVERT FAILED k=$K s=$S" | tee -a "$SUMMARY"; rm -f output/tmp-f16.gguf; continue; }
    rm -f output/tmp-f16.gguf
  fi

  GGUF_ABS="$(cd output && pwd)/${TAG}-q4km.gguf"
  printf 'FROM %s\nPARAMETER num_ctx 4096\nPARAMETER temperature 0.6\n' "$GGUF_ABS" \
    > "output/Modelfile_${TAG}"
  /home/abhijitk20/ollama/bin/ollama rm "$TAG" >/dev/null 2>&1 || true
  /home/abhijitk20/ollama/bin/ollama create "$TAG" -f "output/Modelfile_${TAG}" >/dev/null 2>&1

  .venv/bin/python /tmp/opencode/show.py "$TAG" > "logs/probe_${TAG}.log" 2>&1
  R=$(grep -c "REFUSED" "logs/probe_${TAG}.log" || true)
  C=$(grep -c "complied" "logs/probe_${TAG}.log" || true)
  echo "k=$K s=$S band=$BAND  refused=$R complied=$C" | tee -a "$SUMMARY"

  # free space for the next config
  /home/abhijitk20/ollama/bin/ollama rm "$TAG" >/dev/null 2>&1 || true
  rm -rf "$DIR" "output/${TAG}-q4km.gguf"
done

echo "=== DONE ===" | tee -a "$SUMMARY"
cat "$SUMMARY"
