#!/usr/bin/env bash
# startup.sh — runs on every VM boot.
# Installs dependencies and clones the repo. Does NOT start training.
#
# GPU VMs use the Deep Learning VM image, so NVIDIA drivers are already present.
# No driver installation or reboot needed.

set -euo pipefail
exec > >(tee -a /tmp/rlgames-startup.log) 2>&1

echo "=== rlgames startup: $(date) ==="

_meta() {
  curl -sf "http://metadata.google.internal/computeMetadata/v1/$1" \
    -H "Metadata-Flavor: Google" 2>/dev/null || true
}

WANDB_API_KEY="$(_meta instance/attributes/wandb-api-key)"
HAS_GPU="$(_meta instance/attributes/gpu)"

# First non-system user (UID >= 1000), fallback to root.
SSH_USER=$(getent passwd | awk -F: '$3 >= 1000 {print $1; exit}')
SSH_USER="${SSH_USER:-root}"

REPO_DIR="/opt/rlgames"

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git curl build-essential screen

# ── Install uv ────────────────────────────────────────────────────────────────
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ── Clone repo (idempotent) ───────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/peterhbromley/rlgames.git "$REPO_DIR"
fi
chown -R "$SSH_USER:$SSH_USER" "$REPO_DIR"

# ── Python dependencies ───────────────────────────────────────────────────────
cd "$REPO_DIR/python"
sudo -u "$SSH_USER" uv sync

if [[ "$HAS_GPU" == "true" ]]; then
  sudo -u "$SSH_USER" uv pip install torch --reinstall \
    --index-url https://download.pytorch.org/whl/cu126
fi

# ── WandB ─────────────────────────────────────────────────────────────────────
if [[ -n "$WANDB_API_KEY" ]]; then
  sudo -u "$SSH_USER" "$REPO_DIR/python/.venv/bin/python" \
    -c "import wandb; wandb.login(key='${WANDB_API_KEY}')" \
    || echo "WandB login failed — run 'wandb login' manually."
fi

echo ""
echo "=== rlgames startup complete: $(date) ==="
echo ""
echo "SSH in and start training:"
echo "  cd $REPO_DIR/python"
echo "  uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml"
