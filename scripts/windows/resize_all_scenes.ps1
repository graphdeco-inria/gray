$ErrorActionPreference = "Stop"

$scenes = Get-ChildItem -Path 'data/360_v2', 'data/db', 'data/tandt' -Directory -ErrorAction SilentlyContinue

$i = 0
foreach ($scene in $scenes) {
    $i += 1
    Write-Host "Resizing scene $($scene.FullName) [$i/$($scenes.Count)]"
    python resize.py -s $scene.FullName --yes
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
