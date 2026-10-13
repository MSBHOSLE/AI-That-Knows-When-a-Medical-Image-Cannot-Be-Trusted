#!/usr/bin/env bash
# ==============================================================================
# Setup Environment Script for AI That Knows When a Medical Image Cannot Be Trusted
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  Setting up Environment for Medical AI Reliability Pipeline"
echo "=================================================================="

PYTHON_BIN=""
export PATH="/opt/homebrew/bin:$PATH"

for CANDIDATE in /opt/homebrew/bin/python3.12 python3.12 /opt/homebrew/bin/python3.11 python3.11 /usr/local/bin/python3.12; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v "$CANDIDATE")
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Falling back to default python3: $(which python3)"
    PYTHON_BIN=$(command -v python3)
fi

echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already present at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing required dependencies..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "------------------------------------------------------------------"
echo "Verifying installation:"
python -c "import torch, torchvision, numpy, scipy, sklearn, medmnist; print(f'PyTorch: {torch.__version__} | MPS available: {torch.backends.mps.is_available()} | NumPy: {numpy.__version__} | MedMNIST: {medmnist.__version__}')"

echo "=================================================================="
echo " Environment setup complete! Activate with:"
echo "   source \"$VENV_DIR/bin/activate\""
echo "=================================================================="
