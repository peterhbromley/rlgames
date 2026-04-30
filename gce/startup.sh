#!/usr/bin/env bash
# startup.sh — runs automatically on first VM boot.
# Installs dependencies and clones the repo. Does NOT start training.
#
# After SSHing in, start training manually:
#   cd /opt/rlgames/python
#   uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml

set -euo pipefail
exec > >(tee -a /tmp/rlgames-startup.log) 2>&1

echo "=== rlgames startup: $(date) ==="

# ── Read instance metadata ────────────────────────────────────────────────────
_meta() {
  curl -sf "http://metadata.google.internal/computeMetadata/v1/$1" \
    -H "Metadata-Flavor: Google" 2>/dev/null || true
}

WANDB_API_KEY="$(_meta instance/attributes/wandb-api-key)"
HAS_GPU="$(_meta instance/attributes/gpu)"

# The GCE SSH user is the Google account email prefix, available from metadata.
SSH_USER="$(_meta instance/attributes/ssh-keys | head -1 | cut -d: -f1 || true)"
# Fallback: first non-root user with a home directory.
if [ -z "$SSH_USER" ] || [ "$SSH_USER" = "root" ]; then
  SSH_USER=$(getent passwd | awk -F: '$3 >= 1000 && $3 < 65534 {print $1}' | head -1 || true)
fi
# Final fallback: root.
if [ -z "$SSH_USER" ]; then
  SSH_USER="root"
fi

echo "SSH user : $SSH_USER"

# ── System packages ───────────────────────────────────────────────────────────
echo "Installing system packages..."
apt-get update -qq
apt-get install -y -qq screen git curl build-essential

# ── NVIDIA drivers (GPU only) ─────────────────────────────────────────────────
if [ "$HAS_GPU" = "true" ]; then
  echo "Installing NVIDIA drivers..."
  apt-get install -y -qq ubuntu-drivers-common
  ubuntu-drivers install
  echo "NVIDIA driver installed."
fi

# ── Install uv ────────────────────────────────────────────────────────────────
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ── Clone repo ────────────────────────────────────────────────────────────────
REPO_DIR="/opt/rlgames"
echo "Cloning repo to $REPO_DIR..."
git clone https://github.com/peterhbromley/rlgames.git "$REPO_DIR"
chown -R "$SSH_USER:$SSH_USER" "$REPO_DIR"

# ── Install Python dependencies ───────────────────────────────────────────────
echo "Installing Python dependencies..."
cd "$REPO_DIR/python"
sudo -u "$SSH_USER" uv sync

if [ "$HAS_GPU" = "true" ]; then
  echo "Installing PyTorch with CUDA support..."
  sudo -u "$SSH_USER" uv pip install torch --reinstall --index-url https://download.pytorch.org/whl/cu124
fi
echo "Dependencies installed."

# ── Configure WandB ───────────────────────────────────────────────────────────
if [ -n "$WANDB_API_KEY" ]; then
  echo "Configuring WandB..."
  sudo -u "$SSH_USER" \
    "$REPO_DIR/python/.venv/bin/python" -c \
    "import wandb; wandb.login(key='${WANDB_API_KEY}')" || \
    echo "WandB login failed — run 'wandb login' manually after SSHing in."
fi

echo ""
echo "=== rlgames startup complete: $(date) ==="
echo ""
echo "SSH in and start training:"
echo "  cd /opt/rlgames/python"
echo "  uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml"
