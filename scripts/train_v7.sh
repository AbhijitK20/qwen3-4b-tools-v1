#!/usr/bin/env bash
# v7 tool-repair LoRA, applied ON TOP of the already-uncensored v6.
#
# Why this order (validated against two published writeups):
#   Mainframework/LLM_Safety ablation docs: "Add Jarvis QLoRA after your
#   abliteration". lukey03/Qwen3.5-9B-abliterated: 3 abliteration passes -> 72%
#   answered, pass 4 -> destroyed, pass 3 + QLoRA -> 100%.
#
# v6 already removed refusals (1/20 on the harmful probe). Its weakness is tool
# calling: 86.0% vs v1's 97.5%. Two causes, both addressed here:
#   1. tool share. v1 trained on tok_train_v3 alone; v6 diluted it to 90% with
#      airoboros and compliance rows. Here it is 87%, with a 13% slice of v6
#      data retained purely as a compliance anchor.
#   2. tool volume. v6 saw 8200 tool rows; this sees 6000 in a shorter run, but
#      at a lower loss target on a base that already knows the format.
#
# Compliance is expected to survive because it lives in v6's base weights and
# this LoRA never sees a refusal as a training signal -- it is measured by
# scripts/strict_probe.py afterwards, not assumed.
#
# seq-len pinned to 1024: README records 2048 OOMs on this 6 GB card.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/train_qlora_v2.py \
  --model output/merged-v6 \
  --tok-prefix _v7 --seq-len 1024 --batch 2 --grad-accum 8 \
  --max-steps 400 --lr 1e-4 --seed 3407 --out output/full4b_v7