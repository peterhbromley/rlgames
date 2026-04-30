#!/usr/bin/env bash
# startup.sh — runs automatically on every VM boot.
#
# GPU VMs require a reboot to load the NVIDIA kernel module after installation.
# Flow:
#   Boot 1: install everything (deps, drivers, Python, PyTorch CUDA), then reboot.
#   Boot 2: load the nvidia module, print "startup complete".
#
# CPU VMs complete in a single boot.
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

REPO_DIR="/opt/rlgames"
SETUP_FLAG="/var/lib/rlgames-setup-complete"

# ── GPU second boot: just load the module and exit ───────────────────────────
if [ "$HAS_GPU" = "true" ] && [ -f "$SETUP_FLAG" ]; then
  echo "Post-driver-reboot boot. Loading nvidia module..."
  modprobe nvidia
  echo ""
  echo "=== rlgames startup complete: $(date) ==="
  echo ""
  echo "SSH in and start training:"
  echo "  cd $REPO_DIR/python"
  echo "  uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml"
  exit 0
fi

# ── System packages ───────────────────────────────────────────────────────────
echo "Installing system packages..."
apt-get update -qq
apt-get install -y -qq screen git curl build-essential

# ── NVIDIA drivers (GPU only) ─────────────────────────────────────────────────
# Uses NVIDIA's CUDA repo to ensure driver >= 550, required for CUDA 12.4.
if [ "$HAS_GPU" = "true" ]; then
  echo "Installing NVIDIA drivers..."
  CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING_DEB}" \
    -O "/tmp/${CUDA_KEYRING_DEB}"
  dpkg -i "/tmp/${CUDA_KEYRING_DEB}"
  apt-get update -qq
  apt-get install -y -qq cuda-drivers
  echo "NVIDIA driver installed (reboot required to load)."
fi

# ── Install uv ────────────────────────────────────────────────────────────────
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ── Clone repo ────────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning repo to $REPO_DIR..."
  git clone https://github.com/peterhbromley/rlgames.git "$REPO_DIR"
fi
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

# ── Done ──────────────────────────────────────────────────────────────────────
touch "$SETUP_FLAG"

if [ "$HAS_GPU" = "true" ]; then
  echo ""
  echo "=== rlgames setup complete: $(date) ==="
  echo "Rebooting to load NVIDIA driver..."
  echo "(Tail /tmp/rlgames-startup.log again after reboot for the final startup complete message)"
  reboot
  exit 0
fi

echo ""
echo "=== rlgames startup complete: $(date) ==="
echo ""
echo "SSH in and start training:"
echo "  cd $REPO_DIR/python"
echo "  uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml"
