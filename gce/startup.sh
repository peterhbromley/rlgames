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

# ── Python dependencies ───────────────────────────────────────────────────────
cd "$REPO_DIR/python"
uv sync

if [[ "$HAS_GPU" == "true" ]]; then
  uv pip install torch --reinstall \
    --index-url https://download.pytorch.org/whl/cu126
fi

# ── WandB ─────────────────────────────────────────────────────────────────────
# Write key to /etc/environment so it's available to all users on login.
if [[ -n "$WANDB_API_KEY" ]]; then
  echo "WANDB_API_KEY=${WANDB_API_KEY}" >> /etc/environment
fi

# Make everything accessible to the SSH user (created lazily by GCE on first login).
chmod -R a+rwX "$REPO_DIR"
# uv caches its Python install in the home dir — ensure it's traversable.
chmod a+rx /root
chmod -R a+rX /root/.cache/uv 2>/dev/null || true

echo ""
echo "=== rlgames startup complete: $(date) ==="
echo ""
echo "SSH in and start training:"
echo "  cd $REPO_DIR/python"
echo "  uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml"
