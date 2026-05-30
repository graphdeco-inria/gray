set -e 

uv venv
source .venv/bin/activate

uv pip install "torch==2.9.1" "torchvision==0.24.1" "xformers" --torch-backend=auto

uv sync --frozen --inexact \
  --no-install-package torch \
  --no-install-package torchvision \
  --no-install-package xformers