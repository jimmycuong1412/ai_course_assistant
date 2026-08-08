#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
python -m pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Run the Streamlit app
echo "Starting AI Course Assistant..."
python -m streamlit run app.py
