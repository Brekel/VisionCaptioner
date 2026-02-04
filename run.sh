#!/bin/bash

# 1. Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 2. Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found in $SCRIPT_DIR/venv"
    echo "Please ensure you have created the venv on this Linux machine."
    exit 1
fi

echo "Launching GUI..."

# 3. Start the python GUI script
python main.py
