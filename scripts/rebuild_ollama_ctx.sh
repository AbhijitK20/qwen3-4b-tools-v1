#!/usr/bin/env bash
# Rebuild the serving models with a larger num_ctx than the original 4096 Modelfile.
set -euo pipefail
cd "$(dirname "$0")/.."
CTX="${CTX:-16384}"
BIN="$HOME/ollama/bin/ollama"
export OLLAMA_HOST=127.0.0.1:11434

make_modelfile() {
  .venv/bin/python - "$1" "$CTX" <<'PY'
import pathlib, sys
gguf, ctx = sys.argv[1], sys.argv[2]
pathlib.Path("output/Modelfile.ctx").write_text(
    f"FROM {gguf}\nPARAMETER num_ctx {ctx}\nPARAMETER temperature 0.6\n"
    'TEMPLATE "{{ if .Tools }}system\n{{ . }}\n{{ range .Messages }}{{ .Role }}\n'
    '{{ .Content }}\n{{ end }}\nassistant\n{{ end }}"\n')
PY
}

register() {
  local name="$1" gguf="$2"
  make_modelfile "$gguf"
  "$BIN" create "$name" -f output/Modelfile.ctx >/dev/null 2>&1
  echo "  $name  (num_ctx=$CTX)"
}

./scripts/ollama_up.sh >/dev/null

# v7h already has a Q4 GGUF on disk
register qwen3-4b-v7h "$PWD/output/merged-v7h-q4_k_m.gguf"

# v6: quantise from the merged fp16 export
if [ ! -f output/v6-ctx-q4km.gguf ]; then
  .venv/bin/python repos/llama.cpp/convert_hf_to_gguf.py output/merged-v6 \
    --outfile output/v6-ctx-f16.gguf --outtype f16 >/dev/null 2>&1
  repos/llama.cpp/build/bin/llama-quantize output/v6-ctx-f16.gguf output/v6-ctx-q4km.gguf Q4_K_M 8 >/dev/null 2>&1
  rm -f output/v6-ctx-f16.gguf
fi
register qwen3-4b-v6 "$PWD/output/v6-ctx-q4km.gguf"
