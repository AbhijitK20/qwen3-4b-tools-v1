#!/bin/bash
cd "$HOME/model training"
source .venv/bin/activate
export HF_HOME="$HOME/model training/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "=== hermes-function-calling-v1 (cleaned tool data) ==="
hf download NousResearch/hermes-function-calling-v1 --repo-type dataset || true
echo "=== BFCL eval set ==="
hf download gorilla-llm/BFCL --repo-type dataset || true
echo "=== Qwen3-1.7B backup ==="
hf download unsloth/Qwen3-1.7B-unsloth-bnb-4bit || true
echo "=== EXTRAS DONE ==="
