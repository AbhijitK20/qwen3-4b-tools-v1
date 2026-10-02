#!/bin/bash
set -e
cd ~/"model training"
source .venv/bin/activate
source scripts/env.sh
echo "=== installing torch (CUDA) ==="
uv pip install torch torchvision
echo "=== installing training stack ==="
uv pip install unsloth trl peft bitsandbytes datasets accelerate transformers huggingface_hub hf_transfer
echo "=== versions ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import unsloth, transformers, trl, peft, bitsandbytes; print('unsloth', unsloth.__version__); print('transformers', transformers.__version__); print('trl', trl.__version__); print('peft', peft.__version__); print('bitsandbytes', bitsandbytes.__version__)"
