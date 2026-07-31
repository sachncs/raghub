#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "=== raghub cleanup ==="
echo ""
echo "This will remove:"
echo "  - .venv/  (virtual environment)"
echo "  - __pycache__/ directories and *.pyc files"
echo "  - data/  (all runtime data: SQLite DBs, image store, ZVec index)"
echo "  - *.egg-info/  (package metadata)"
echo "  - .pytest_cache/"
echo "  - *.pyc, *.pyo, *.~ files"
echo ""

read -rp "Continue? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# Deactivate venv if active
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "  Deactivating virtual environment ..."
    deactivate 2>/dev/null || true
fi

echo "[1/5] Removing virtual environment ..."
rm -rf .venv
echo "  OK"

echo "[2/5] Removing __pycache__ and .pyc files ..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  OK"

echo "[3/5] Removing runtime data ..."
rm -rf data
echo "  OK"

echo "[4/5] Removing egg-info and build artifacts ..."
rm -rf *.egg-info .pytest_cache
echo "  OK"

echo "[5/5] Removing backup files ..."
find . -type f -name "*.~" -delete 2>/dev/null || true
echo "  OK"

echo ""
echo "Cleanup complete. Run ./setup.sh to rebuild."
