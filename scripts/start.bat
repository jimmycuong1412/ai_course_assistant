@echo off
setlocal enabledelayedexpansion

REM Change directory to the project root (parent directory of scripts/)
cd /d "%~dp0\.."

REM Check if uv is available
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Detected 'uv'. Using uv for virtual environment and execution...
    
    if not exist ".venv" (
        echo Creating virtual environment with uv...
        uv venv
    )
    
    echo Installing dependencies with uv...
    uv pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies with uv
        pause
        exit /b 1
    )
    
    echo Starting AI Course Assistant...
    set PYTHONPATH=.
    uv run streamlit run src/app.py --server.fileWatcherType none
    goto end
)

REM Fallback: Standard Python execution
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Neither 'uv' nor 'python' is installed or in PATH
    pause
    exit /b 1
)

echo [INFO] 'uv' not found. Using standard Python...
echo Installing dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo Starting AI Course Assistant...
set PYTHONPATH=.
python -m streamlit run src/app.py --server.fileWatcherType none

:end
pause