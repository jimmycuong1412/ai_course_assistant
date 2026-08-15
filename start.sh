#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "[INFO] Detected 'uv'. Using uv for virtual environment and execution..."
    
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment with uv..."
        uv venv
    fi
    
    echo "Installing dependencies with uv..."
    uv pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies with uv"
        exit 1
    fi
    
    echo "Starting AI Course Assistant..."
    uv run streamlit run app.py --server.fileWatcherType none
    exit 0
fi

# Fallback: Standard Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Neither 'uv' nor 'python' is installed or in PATH"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo "[INFO] 'uv' not found. Using standard Python: $($PYTHON_CMD --version)..."

echo "Installing dependencies..."
$PYTHON_CMD -m pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo "Starting AI Course Assistant..."
$PYTHON_CMD -m streamlit run app.py --server.fileWatcherType none