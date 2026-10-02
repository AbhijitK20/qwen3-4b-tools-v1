#!/bin/bash
cd "$HOME/model training"
source .venv/bin/activate
export HF_HOME="$HOME/model training/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "=== 8B ==="
hf download unsloth/Qwen3-8B-unsloth-bnb-4bit
echo "=== 1.7B ==="
hf download unsloth/Qwen3-1.7B-unsloth-bnb-4bit
echo "=== MODELS DONE ==="
