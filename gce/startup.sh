#!/usr/bin/env bash
# startup.sh — runs automatically on first VM boot.
# Sets up the environment and starts training as the SSH user.
#
# Monitor progress after SSHing in:
#   tail -f /tmp/rlgames-training.log
#   screen -r training

set -euo pipefail
exec > >(tee -a /tmp/rlgames-startup.log) 2>&1

echo "=== rlgames startup: $(date) ==="

# ── Read instance metadata ────────────────────────────────────────────────────
_meta() {
  curl -sf "http://metadata.google.internal/computeMetadata/v1/$1" \
    -H "Metadata-Flavor: Google" 2>/dev/null || true
}

TRAINING_CONFIG="$(_meta instance/attributes/training-config)"
TRAINING_CONFIG="${TRAINING_CONFIG:-training/configs/oh_hell_ppo_variable.yaml}"
WANDB_API_KEY="$(_meta instance/attributes/wandb-api-key)"

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

if [ "$SSH_USER" = "root" ]; then
  USER_HOME="/root"
else
  USER_HOME="/home/$SSH_USER"
fi

echo "SSH user        : $SSH_USER"
echo "Training config : $TRAINING_CONFIG"

# ── System packages ───────────────────────────────────────────────────────────
echo "Installing system packages..."
apt-get update -qq
apt-get install -y -qq screen git curl build-essential

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
echo "Dependencies installed."

# ── Configure WandB ───────────────────────────────────────────────────────────
if [ -n "$WANDB_API_KEY" ]; then
  echo "Configuring WandB..."
  sudo -u "$SSH_USER" \
    "$REPO_DIR/python/.venv/bin/python" -c \
    "import wandb; wandb.login(key='${WANDB_API_KEY}')" || \
    echo "WandB login failed — run 'wandb login' manually after SSHing in."
fi

# ── Write convenience start script ───────────────────────────────────────────
TRAIN_CMD="cd $REPO_DIR/python && $REPO_DIR/python/.venv/bin/python -m training.train_ppo --config $TRAINING_CONFIG"
TRAIN_LOG="/tmp/rlgames-training.log"

cat > "$USER_HOME/start_training.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail
if screen -list 2>/dev/null | grep -q "training"; then
  echo "Training already running. Attach with: screen -r training"
else
  echo "Starting training..."
  screen -dmS training bash -c '${TRAIN_CMD} 2>&1 | tee -a ${TRAIN_LOG}; exec bash'
  echo "Training started. Attach with: screen -r training"
  echo "Logs: tail -f ${TRAIN_LOG}"
fi
EOF
chmod +x "$USER_HOME/start_training.sh"
chown "$SSH_USER:$SSH_USER" "$USER_HOME/start_training.sh"

# ── Start training ────────────────────────────────────────────────────────────
echo "Starting training as $SSH_USER..."
sudo -u "$SSH_USER" screen -dmS training \
  bash -c "${TRAIN_CMD} 2>&1 | tee -a ${TRAIN_LOG}; exec bash"

echo ""
echo "=== rlgames startup complete: $(date) ==="
echo ""
echo "SSH in and check training:"
echo "  tail -f /tmp/rlgames-training.log"
echo "  screen -r training"
