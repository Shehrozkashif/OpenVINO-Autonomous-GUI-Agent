#!/bin/bash
# Launch the Desktop GUI Agent
# Sets LD_LIBRARY_PATH for libxcb-cursor (extracted without root) and activates venv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$HOME/.local_xcb/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

source "$SCRIPT_DIR/venv/bin/activate"
exec python "$SCRIPT_DIR/main.py" "$@"
