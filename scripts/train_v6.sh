#!/usr/bin/env bash
# v6 LoRA. Two fixes over v5, which FAILED (BFCL call rate 26.5%, worst measured):
#   1. tool-calling slice from tok_train_v3 (91% abstention), not v4 (75%)
#   2. tool share 90% (v5: 76%) -- the 22% prose airoboros share is what pushed
#      v5 toward answering instead of calling
# Compliance slice is 490 rows vs v5's 192, so the refusal signal survives the cut.
# seq-len pinned to 1024: README records 2048 OOMs on this 6 GB card.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/train_qlora_v2.py \
  --tok-prefix _v6_full --seq-len 1024 --batch 2 --grad-accum 8 \
  --max-steps 700 --lr 2e-4 --seed 3407 --out output/full4b_v6
