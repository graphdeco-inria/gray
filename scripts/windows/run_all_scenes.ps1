$ErrorActionPreference = "Stop"

if ($args.Count -eq 0) {
    Write-Error "Usage: .\\scripts\\windows\\run_all_scenes.ps1 <output-root> [args...]"
}

$outputRoot = $args[0]
$extraArgs = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
$runScript = (Resolve-Path (Join-Path $PSScriptRoot '..\..\windows\run.ps1')).Path

function Invoke-GrayRun {
    param(
        [string]$ModelPath,
        [string]$Resize,
        [string]$ScenePath
    )

    $runArgs = @($ModelPath, '-r', $Resize, '-s', $ScenePath, '--eval') + $extraArgs
    & $runScript @runArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

foreach ($scene in @('treehill', 'flowers', 'garden', 'stump', 'bicycle')) {
    Invoke-GrayRun -ModelPath (Join-Path $outputRoot $scene) -Resize '4' -ScenePath "data/360_v2/$scene"
}

foreach ($scene in @('kitchen', 'room', 'bonsai', 'counter')) {
    Invoke-GrayRun -ModelPath (Join-Path $outputRoot $scene) -Resize '2' -ScenePath "data/360_v2/$scene"
}

foreach ($root in @('data/db', 'data/tandt')) {
    foreach ($scene in Get-ChildItem -Path $root -Directory | Sort-Object Name) {
        $scenePath = Resolve-Path -Relative $scene.FullName
        Invoke-GrayRun -ModelPath (Join-Path $outputRoot $scene.Name) -Resize '1' -ScenePath $scenePath
    }
}
