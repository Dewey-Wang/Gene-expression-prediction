#!/usr/bin/env bash
set -euo pipefail

# ---------- OS guard ----------
OS="$(uname -s)"
case "$OS" in
  Linux*)  MACHINE=Linux ;;
  Darwin*) MACHINE=Mac ;;
  CYGWIN*|MINGW*|MSYS*) MACHINE=Windows ;;
  *)       MACHINE="UNKNOWN:$OS" ;;
esac

if [[ "$MACHINE" == "Windows" ]]; then
  echo "❌ Native Windows shells (CYGWIN/MINGW/MSYS) are not supported."
  echo "👉 Please run this in WSL2 (Ubuntu) or use Docker."
  exit 1
fi

# ---------- Python check ----------
PYTHON_BIN="python3"
REQUIRED="3.12"
CURRENT="$($PYTHON_BIN -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"

# Require >= 3.12
if [[ "$(printf '%s\n' "$REQUIRED" "$CURRENT" | sort -V | head -n1)" != "$REQUIRED" ]]; then
  echo "❌ Python >= $REQUIRED required. Current: $CURRENT"
  exit 1
fi

# ---------- venv setup ----------
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------- pip & deps ----------
python -m pip install --upgrade pip
REQ="requirements.txt"
if [[ ! -f "$REQ" ]]; then
  echo "❌ $REQ not found. Create it first."
  exit 1
fi
python -m pip install -r "$REQ"

# ---------- (optional) Jupyter kernel ----------
# comment out if you don't need a kernel

python -m ipykernel install --user --name=ml4g_project1 --display-name "Python (.venv) ml4g_project1" || true

echo "✅ Environment ready."
echo "👉 Activate: source .venv/bin/activate"
