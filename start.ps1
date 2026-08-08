# AI Course Assistant Startup Script
# Run this script to start the application

param(
    [switch]$NoInstall
)

# Get script directory
$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion"
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Install dependencies unless --NoInstall is specified
if (-not $NoInstall) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    python -m pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Run the Streamlit app
Write-Host "Starting AI Course Assistant..." -ForegroundColor Green
Write-Host "Opening http://localhost:8501 in your browser..." -ForegroundColor Green
python -m streamlit run app.py
