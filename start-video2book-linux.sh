#!/bin/bash
# Start Video2Book (double-click, or run ./start-video2book-linux.sh). The first start installs it (needs internet once).
cd "$(dirname "$0")" || exit 1
PY=python3
if [ ! -x .venv/bin/python ]; then
  if ! command -v "$PY" >/dev/null 2>&1 || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is needed: e.g. sudo apt install python3 python3-venv"
    read -r -p "Press Enter to close."
    exit 1
  fi
  echo "First start: installing Video2Book. This takes a few minutes..."
  if ! ("$PY" -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -e ".[ocr]"); then
    echo "Installation failed - see the messages above. Delete the .venv folder and try again."
    read -r -p "Press Enter to close."
    exit 1
  fi
fi
exec .venv/bin/python -m video2book.launch
