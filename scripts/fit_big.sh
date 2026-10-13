#!/usr/bin/env bash
# Refit the same-prompt refusal subspace at higher rank and cache it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts

K="${1:-300}"
.venv/bin/python - "$K" <<'PY'
import sys, torch, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from uncensor import fit_subspace
k = int(sys.argv[1])
d = 'output/merged-qwen3_4b_s1337'
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16,
        device_map='cpu', low_cpu_mem_usage=True).eval()
P = fit_subspace(m, tok, 'cpu', 510, k, 0, contrast='same')
out = pathlib.Path(d)/'_subspace_big.pt'
torch.save(P, out)
print('SAVED', tuple(P.shape), '->', out)
PY
