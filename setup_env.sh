#!/usr/bin/env bash
# Setup mmpose environment using uv
# Usage: ./setup_env.sh
#
# NOTE: Legacy OpenMMLab checkpoints require torch.load(weights_only=False)
# because they contain numpy arrays. PyTorch 2.6+ defaults to weights_only=True.
# When training, patch torch.load before importing mmengine/mmpose, or re-save
# checkpoints with: torch.save(torch.load(path, weights_only=False), path)
set -euo pipefail

VENV=".venv"
PY="$VENV/bin/python"

echo "==> Creating venv and installing dependencies..."
uv sync -p python3.11

echo "==> Pinning setuptools (mmcv needs pkg_resources, removed in setuptools>=70)..."
uv pip install --python "$PY" "setuptools<70" wheel

echo "==> Installing mmcv (building with CUDA ops from source)..."
CUDA_HOME=/usr/local/cuda-12 FORCE_CUDA=1 uv pip install --python "$PY" "mmcv>=2.0.0,<2.2.0" --no-build-isolation

echo "==> Installing mmpose in editable mode..."
uv pip install --python "$PY" -e . --no-build-isolation

echo "==> Re-saving checkpoints for PyTorch 2.6+ compatibility..."
$PY -c "
import torch, glob, os
for f in glob.glob('checkpoints/*.pth'):
    try:
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        torch.save(ckpt, f)
        print(f'  Re-saved: {f}')
    except Exception as e:
        print(f'  Skipped {f}: {e}')
" 2>/dev/null || true

echo "==> Verifying installation..."
$PY -c "
import torch
print(f'  PyTorch:     {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import mmcv;      print(f'  mmcv:        {mmcv.__version__}')
import mmengine;  print(f'  mmengine:    {mmengine.__version__}')
import mmpose;    print(f'  mmpose:      {mmpose.__version__}')
import mmdet;     print(f'  mmdet:       {mmdet.__version__}')
import numpy;     print(f'  numpy:       {numpy.__version__}')
import onnxruntime; print(f'  onnxruntime: {onnxruntime.__version__}')
from mmpose.apis import init_model
from mmpose.engine.hooks import FreezeBackboneHook
print('  All imports OK!')
"

echo "==> Done! Activate with: source .venv/bin/activate"
