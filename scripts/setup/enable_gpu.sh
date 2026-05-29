#!/bin/bash
# scripts/setup/enable_gpu.sh
# Enables Flash Attention for Ollama — allows qwen2.5vl to run on GPU.
#
# Problem: qwen2.5vl's vision compute graph needs 7.3 GiB VRAM.
#          RTX 2060 has 5.6 GiB free. Without Flash Attention: falls back to CPU.
#          With Flash Attention: compute graph shrinks to ~1 GiB → fits on GPU.
#
# Run with: sudo bash scripts/setup/enable_gpu.sh

set -e

echo "==> Creating Ollama service override for Flash Attention..."
mkdir -p /etc/systemd/system/ollama.service.d/
cat > /etc/systemd/system/ollama.service.d/flash_attention.conf << 'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

echo "==> Reloading systemd and restarting Ollama..."
systemctl daemon-reload
systemctl restart ollama

echo "==> Waiting for Ollama to start..."
sleep 3

echo "==> Verifying Flash Attention is active..."
journalctl -u ollama --since "1 minute ago" --no-pager | grep -i "flash" || echo "  (check: ollama ps after loading a model)"

echo ""
echo "Done. Now run: python e2e_verify.py"
