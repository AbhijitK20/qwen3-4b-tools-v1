#!/bin/bash
cd ~/"model training"
source .venv/bin/activate
export HF_HOME=~/"model training"/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "=== [1/8] Qwen3-8B 4-bit (QLoRA base) ==="
hf download unsloth/Qwen3-8B-unsloth-bnb-4bit
echo "=== [2/8] Qwen3-1.7B 4-bit (backup) ==="
hf download unsloth/Qwen3-1.7B-unsloth-bnb-4bit || hf download unsloth/Qwen3-1.7B-bnb-4bit
echo "=== [3/8] xLAM function calling 60k ==="
hf download Salesforce/xlam-function-calling-60k --repo-type dataset
echo "=== [4/8] glaive function calling v2 ==="
hf download glaiveai/glaive-function-calling-v2 --repo-type dataset
echo "=== [5/8] ToolACE ==="
hf download Team-ACE/ToolACE --repo-type dataset || true
echo "=== [6/8] SWE-Gym ==="
hf download SWE-Gym/SWE-Gym --repo-type dataset || true
echo "=== [7/8] When2Call ==="
hf download nvidia/When2Call --repo-type dataset || true
echo "=== [8/8] ToolBench sample ==="
hf download microsoft/ToolBench --repo-type dataset || true
echo "=== ALL DOWNLOADS DONE ==="
du -sh "$HF_HOME"
