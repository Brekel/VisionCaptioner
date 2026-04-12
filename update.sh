#!/bin/bash

# 1. Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo
echo "------------------------------------------"
echo "      UPDATING VISION CAPTIONER"
echo "------------------------------------------"
echo

echo "[1/3] Pulling latest changes from git..."
if ! git pull; then
    echo
    echo "WARNING: git pull failed. Continuing anyway..."
    echo
fi

echo
echo "[2/3] Activating environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: venv not found at $SCRIPT_DIR/venv"
    echo "Please create the virtual environment first."
    exit 1
fi

echo
echo "[3/3] Upgrading pip packages from requirements.txt..."
python -m pip install --upgrade pip
if ! python -m pip install --upgrade -r requirements.txt; then
    echo
    echo "ERROR: pip install failed. See messages above."
    exit 1
fi

echo
echo "------------------------------------------"
echo "      UPDATE COMPLETE"
echo "------------------------------------------"
echo
