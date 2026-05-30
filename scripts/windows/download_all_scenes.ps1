$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path data | Out-Null

Write-Host 'Downloading scenes to `data/`...'

curl.exe -fL --progress-bar -o data/360_v2.zip https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m zipfile -e data/360_v2.zip data/360_v2/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item data/360_v2.zip
Remove-Item data/360_v2/flowers.txt
Remove-Item data/360_v2/treehill.txt

curl.exe -fL --progress-bar -o data/360_extra_scenes.zip https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m zipfile -e data/360_extra_scenes.zip data/360_v2/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item data/360_extra_scenes.zip

curl.exe -fL --progress-bar -o data/tandt_db.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m zipfile -e data/tandt_db.zip data/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item data/tandt_db.zip

Write-Host 'All scenes downloaded.'
