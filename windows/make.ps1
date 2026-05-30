$ErrorActionPreference = "Stop"

cmake -S . -B build -G "Ninja" @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build build --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }