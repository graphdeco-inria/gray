$ErrorActionPreference = "Stop"

uv venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

. .\.venv\Scripts\Activate.ps1

uv pip install `
  "torch==2.9.1" `
  "torchvision==0.24.1" `
  "xformers" `
  "git+https://github.com/rahul-goel/fused-ssim/" `
  --torch-backend=auto
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv sync --frozen --inexact `
  --no-install-package torch `
  --no-install-package torchvision `
  --no-install-package xformers `
  --no-install-package pycolmap-cuda12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }