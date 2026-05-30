$ErrorActionPreference = "Stop"

if ($args.Count -eq 0) {
    Write-Error "Usage: .\\windows\\run.ps1 <model-path> [args...]"
}

python train.py --eval -m @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$modelPath = $args[0]

python render.py -m $modelPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python metrics.py -m $modelPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python measure_fps.py -m $modelPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
