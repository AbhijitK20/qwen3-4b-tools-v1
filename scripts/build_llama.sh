#!/bin/bash
cd ~/"model training"
source .venv/bin/activate
echo "=== llama.cpp python deps ==="
uv pip install -r repos/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
echo "=== building llama.cpp (CPU build for quantize) ==="
cd repos/llama.cpp
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
cmake --build build --config Release -j12
echo "=== built binaries ==="
ls build/bin/ | grep -E "quantize|llama-cli" 
