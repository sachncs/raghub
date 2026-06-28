#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "=== raghub development setup ==="
echo ""

# ---------- OS detection ----------
OS="$(uname -s)"
case "$OS" in
    Darwin)
        PKG_MGR="brew"
        ;;
    Linux)
        PKG_MGR="apt"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# ---------- system dependencies ----------
if ! command -v "$PKG_MGR" &>/dev/null; then
    echo "ERROR: $PKG_MGR not found. Please install it first."
    exit 1
fi

echo "[1/6] Installing system dependencies ..."

if [ "$PKG_MGR" = "brew" ]; then
    brew install tesseract libmagic 2>/dev/null || true
elif [ "$PKG_MGR" = "apt" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq tesseract-ocr libmagic1 2>/dev/null || true
fi

echo "  System dependencies: OK"

# ---------- Python check ----------
echo "[2/6] Checking Python ..."
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1)
        MAJOR=$(echo "$VER" | sed -n 's/.* \([0-9]*\)\.[0-9]*\.[0-9]*.*/\1/p; s/.* \([0-9]*\)\.[0-9]*.*/\1/p')
        MINOR=$(echo "$VER" | sed -n 's/.* [0-9]*\.\([0-9]*\)\..*/\1/p; s/.* [0-9]*\.\([0-9]*\).*/\1/p')
        if [ -z "$MAJOR" ]; then
            continue
        fi
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required. Install it and try again."
    exit 1
fi
echo "  Using: $($PYTHON --version)"

# ---------- virtual environment ----------
echo "[3/6] Creating virtual environment ..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping creation."
else
    $PYTHON -m venv .venv
    echo "  .venv created."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "  Python: $(which python)"
echo "  Pip:    $(pip --version 2>&1 | head -1)"

# ---------- install pip dependencies ----------
echo "[4/6] Installing pip packages (this may take a minute) ..."
pip install --quiet --upgrade pip 2>/dev/null
if [ -f requirements/all.txt ]; then
    pip install --quiet -r requirements/all.txt
else
    if [ -d requirements ]; then
        for req_file in requirements/*.txt; do
            [ -f "$req_file" ] && pip install --quiet -r "$req_file"
        done
    fi
fi
echo "  pip packages: OK"

# ---------- editable install ----------
echo "[5/6] Installing raghub in editable mode ..."
pip install --quiet -e .
echo "  editable install: OK"

# ---------- data directories ----------
echo "[6/6] Creating data directories ..."
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
echo "    python -m pytest tests/ -v"
echo ""
