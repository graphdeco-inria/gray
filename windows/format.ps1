$ErrorActionPreference = "Stop"

$clangFiles = Get-ChildItem -Path cuda -Recurse -File | Where-Object {
    $_.FullName -notmatch '[\\/]third_party[\\/]' -and $_.Extension -in @('.cpp', '.cuh', '.cu', '.h')
} | Sort-Object FullName

foreach ($file in $clangFiles) {
    clang-format -i --style='{SortIncludes: false, ColumnLimit: 120, IndentWidth: 4, UseTab: Never, IncludeBlocks: Preserve}' $file.FullName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$ruffTargets = @(
    Get-ChildItem -Path . -Filter *.py -File | Sort-Object Name | ForEach-Object { $_.FullName }
)
$ruffTargets += 'gray'

ruff format @ruffTargets
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
