$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Setup conda, then activate the image-stitcher environment
Write-Host "Setting up conda..."
conda init powershell
conda shell.powershell hook | Out-String | Invoke-Expression

Write-Host "Activating the image-stitcher conda environment..."
conda activate image-stitcher

Write-Host "Running the image stitcher gui..."
python -m image_stitcher.stitcher_gui
