#!/bin/bash
# scripts/setup/start_dev.sh
# Development startup sequence for AMD CPU + RTX 2060.
# Run this every session before launching main.py.

set -e
cd "$(dirname "$0")/../.."

echo "=== Desktop GUI Agent — Dev Startup ==="
echo ""

# 1. X11 check
if [ "$XDG_SESSION_TYPE" = "wayland" ]; then
    echo "[FAIL] Wayland detected — pyautogui won't work."
    echo "       Add to ~/.bashrc: export GDK_BACKEND=x11"
    exit 1
fi
echo "[OK] Session: $XDG_SESSION_TYPE  DISPLAY=$DISPLAY"

# 2. Ollama
echo ""
echo "==> Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "[FAIL] Ollama not installed. Install from https://ollama.com"
    exit 1
fi

if ! ollama list &>/dev/null; then
    echo "  Starting ollama serve in background..."
    ollama serve &>/tmp/ollama.log &
    sleep 3
fi

# Check qwen2.5vl model
if ollama list | grep -q "qwen2.5vl:3b"; then
    echo "[OK] qwen2.5vl:3b found"
else
    echo "[WARN] qwen2.5vl:3b not pulled yet."
    echo "       Pulling now (~2 GB)..."
    ollama pull qwen2.5vl:3b
    echo "[OK] qwen2.5vl:3b pulled"
fi

# 3. Tool server
echo ""
echo "==> Starting Tool Server on port 8015..."
if curl -s http://127.0.0.1:8015/health &>/dev/null; then
    echo "[OK] Tool server already running"
else
    python -m tools.desktop_control.server &>/tmp/tool_server.log &
    sleep 2
    if curl -s http://127.0.0.1:8015/health &>/dev/null; then
        echo "[OK] Tool server started"
    else
        echo "[FAIL] Tool server failed to start — check /tmp/tool_server.log"
        exit 1
    fi
fi

echo ""
echo "=== All systems ready ==="
echo ""
echo "Launch the app:  python main.py"
echo "Pipeline:        Ollama (Dev — qwen2.5vl)"
echo ""
