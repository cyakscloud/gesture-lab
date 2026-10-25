$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host 'Building a private Python environment for PALM//PILOT…' -ForegroundColor Cyan
    python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python could not create the project environment.' }
}

$ready = $false
if (Test-Path -LiteralPath $python) {
    & $python -c 'import cv2, mediapipe' 2>$null
    $ready = ($LASTEXITCODE -eq 0)
}
if (-not $ready) {
    Write-Host 'Installing the camera + gesture stack (one-time setup)…' -ForegroundColor Cyan
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip inside the project environment.' }
    & $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the error above and try again.' }
}

& $python (Join-Path $PSScriptRoot 'gesture_lab.py') @args
exit $LASTEXITCODE
