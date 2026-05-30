$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path output/pretrained | Out-Null

Write-Host 'Downloading scenes to `output/pretrained`...'

$scenes = @('bicycle', 'bonsai', 'counter', 'drjohnson', 'flowers', 'garden', 'kitchen', 'playroom', 'room', 'stump', 'train', 'treehill', 'truck')

foreach ($scene in $scenes) {
    $archivePath = "output/pretrained/$scene.zip"
    curl.exe -fL --progress-bar -o $archivePath "https://repo-sam.inria.fr/nerphys/gray/pretrained/$scene.zip"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    python -m zipfile -e $archivePath output/pretrained/
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Remove-Item $archivePath
}
