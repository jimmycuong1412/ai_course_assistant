# AI Course Assistant Startup Script
param(
    [switch]$NoInstall
)

# Get project root directory (parent directory of scripts/)
$projectRoot = Split-Path -Parent -Path (Split-Path -Parent -Path $MyInvocation.MyCommand.Definition)
Set-Location $projectRoot

# Check if uv is available
$hasUv = Get-Command uv -ErrorAction SilentlyContinue

if ($hasUv) {
    Write-Host "[INFO] Detected 'uv'. Managing environment with uv..." -ForegroundColor Cyan
    
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating virtual environment with uv..." -ForegroundColor Cyan
        uv venv
    }
    
    if (-not $NoInstall) {
        Write-Host "Installing dependencies with uv..." -ForegroundColor Cyan
        uv pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to install dependencies with uv" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "Starting AI Course Assistant..." -ForegroundColor Green
    Write-Host "Opening http://localhost:8501 in your browser..." -ForegroundColor Green
    $env:PYTHONPATH = "."
    uv run streamlit run src/app.py --server.fileWatcherType none
    exit 0
}

# Fallback: Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[INFO] 'uv' not found. Found: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Error: Neither 'uv' nor 'python' is installed or in PATH" -ForegroundColor Red
    exit 1
}

if (-not $NoInstall) {
    Write-Host "Installing dependencies with pip..." -ForegroundColor Cyan
    python -m pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Starting AI Course Assistant..." -ForegroundColor Green
Write-Host "Opening http://localhost:8501 in your browser..." -ForegroundColor Green
$env:PYTHONPATH = "."
python -m streamlit run src/app.py --server.fileWatcherType none