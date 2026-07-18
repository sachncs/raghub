#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "=== raghub development setup ==="
echo ""

OS="$(uname -s)"
case "$OS" in
    Darwin) PKG_MGR="brew" ;;
    Linux)  PKG_MGR="apt" ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

if ! command -v "$PKG_MGR" &>/dev/null; then
    echo "ERROR: $PKG_MGR not found. Please install it first."
    exit 1
fi

echo "[1/5] Installing system dependencies ..."
if [ "$PKG_MGR" = "brew" ]; then
    brew install tesseract libmagic 2>/dev/null || true
else
    sudo apt-get update -qq
    sudo apt-get install -y -qq tesseract-ocr libmagic1 2>/dev/null || true
fi
echo "  System dependencies: OK"

echo "[2/5] Locating Python 3.12 ..."
PYTHON=""
for cmd in python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12 is required. Install it and re-run setup.sh."
    echo "  macOS:  brew install python@3.12"
    echo "  Linux:  sudo apt-get install python3.12 python3.12-venv"
    exit 1
fi
echo "  Using: $($PYTHON --version)"

echo "[3/5] Creating virtual environment ..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping creation."
else
    "$PYTHON" -m venv .venv
    echo "  .venv created."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "  Python: $(python --version)"
echo "  Pip:    $(pip --version 2>&1 | head -1)"

echo "[4/5] Installing project + extras ..."
pip install --quiet --upgrade pip 2>/dev/null
pip install --quiet -e ".[dev,api,ui]"
echo "  pip packages: OK"

echo "[5/5] Creating data directories ..."
mkdir -p data
echo "  data/: OK"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  Activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  Run tests:"
echo "    python -m pytest tests/"
echo ""
