#!/bin/bash
cd ~/"model training"
source .venv/bin/activate
export HF_HOME=~/"model training"/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
for item in "unsloth/Qwen3-1.7B-unsloth-bnb-4bit|model" \
            "Salesforce/xlam-function-calling-60k|dataset" \
            "Team-ACE/ToolACE|dataset" \
            "SWE-Gym/SWE-Gym|dataset" \
            "microsoft/ToolBench|dataset"; do
  repo="${item%%|*}"; kind="${item##*|}"
  echo "=== downloading $repo ($kind) ==="
  for attempt in 1 2 3; do
    if hf download "$repo" --repo-type "$kind"; then echo "OK $repo"; break; fi
    echo "retry $attempt for $repo"; sleep 5
  done
done
echo "=== DONE ==="
