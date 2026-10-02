#!/bin/bash
cd "$HOME/model training"
source .venv/bin/activate
echo "=== reinstalling CUDA torch (cu130) ==="
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -3
echo "=== verify ==="
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
