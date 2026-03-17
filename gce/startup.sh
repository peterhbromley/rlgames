#!/usr/bin/env bash
# startup.sh — runs automatically on first VM boot.
# Clones the repo, installs dependencies, and verifies the GPU is accessible.
#
# Logs are written to /var/log/rlgames-startup.log so you can check progress:
#   sudo journalctl -u google-startup-scripts -f
#   tail -f /var/log/rlgames-startup.log

set -euo pipefail
export HOME=/root
exec > >(tee -a /tmp/rlgames-startup.log) 2>&1

echo "=== rlgames startup: $(date) ==="

# ── Wait for NVIDIA driver (installed by Deep Learning VM image on first boot) ──
echo "Waiting for NVIDIA driver..."
for i in $(seq 1 30); do
    if nvidia-smi > /dev/null 2>&1; then
        echo "NVIDIA driver ready."
        break
    fi
    echo "  attempt $i/30, retrying in 10s..."
    sleep 10
done
nvidia-smi

# ── Install uv ────────────────────────────────────────────────────────────────
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ── Clone repo ────────────────────────────────────────────────────────────────
REPO_DIR="/opt/rlgames"
echo "Cloning repo to $REPO_DIR..."
git clone https://github.com/peterhbromley/rlgames.git "$REPO_DIR"

# ── Install Python dependencies ───────────────────────────────────────────────
echo "Installing Python dependencies..."
cd "$REPO_DIR/python"
uv sync

# ── Verify CUDA is visible from PyTorch ──────────────────────────────────────
echo "Verifying PyTorch CUDA access..."
uv run python -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
"

echo "=== rlgames startup complete: $(date) ==="
echo "To start training, SSH in and run:"
echo "  cd $REPO_DIR/python"
echo "  uv run python -m training.train --config training/configs/oh_hell_full.yaml"
