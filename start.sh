#!/usr/bin/env bash
# ============================================================
#  One-click launcher for the Smartphone Statistics Dashboard
#  macOS / Linux. First run sets everything up; later runs
#  start instantly. Open with:  ./start.sh
# ============================================================
set -e
cd "$(dirname "$0")"

# --- Find Python ---
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python was not found. Install Python 3.10+ from https://www.python.org/downloads/"
  exit 1
fi

# --- Create the virtual environment (first run only) ---
if [ ! -d .venv ]; then
  echo "[1/3] Creating a private Python environment (first run only)..."
  $PY -m venv .venv
fi

# --- Install dependencies ---
echo "[2/3] Installing dependencies..."
.venv/bin/python -m pip install --quiet -r requirements.txt

# --- Start the dashboard ---
echo "[3/3] Starting the dashboard: http://localhost:8501"
echo "(Press Ctrl+C to stop.)"
.venv/bin/python -m streamlit run app.py
