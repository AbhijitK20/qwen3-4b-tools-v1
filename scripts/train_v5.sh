#!/usr/bin/env bash
# v5 LoRA: tool-calling (76%) + airoboros (22%) + distilled compliance (2%).
# seq-len pinned to 1024: README records that 2048 OOMs on this 6 GB card and
# the v3 run died at step 200 with it.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/train_qlora_v2.py \
  --tok-prefix _v5_full \
  --seq-len 1024 \
  --batch 2 \
  --grad-accum 8 \
  --max-steps 700 \
  --lr 2e-4 \
  --seed 3407 \
  --out output/full4b_v5
