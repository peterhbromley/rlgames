#!/usr/bin/env bash
# create_vm.sh — provision a GCE CPU-only Spot VM for RL training.
#
# Prerequisites (run once on your local machine):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   ZONE=us-central1-a MACHINE_TYPE=e2-highcpu-16 ./gce/create_vm.sh
#
# Optional environment variables:
#   ZONE             — GCE zone (required)
#   MACHINE_TYPE     — GCE machine type (default: e2-highcpu-16)
#   WANDB_API_KEY    — forwarded to the VM for experiment tracking
#   TRAINING_CONFIG  — config path relative to python/ (default: training/configs/oh_hell_ppo_variable.yaml)
#   INSTANCE_NAME    — override VM name (default: rlgames-trainer)
#
# To delete the VM when done:
#   gcloud compute instances delete rlgames-trainer --zone=ZONE --quiet

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="${GCLOUD_PROJECT:-$(gcloud config get-value project)}"
INSTANCE_NAME="${INSTANCE_NAME:-rlgames-trainer}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-highcpu-16}"
BOOT_DISK_SIZE="50GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

TRAINING_CONFIG="${TRAINING_CONFIG:-training/configs/oh_hell_ppo_variable.yaml}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ZONE:-}" ]; then
  echo "Error: ZONE is required. Example:"
  echo "  ZONE=us-central1-a ./gce/create_vm.sh"
  exit 1
fi

echo "Project       : $PROJECT"
echo "Zone          : $ZONE"
echo "Machine type  : $MACHINE_TYPE"
echo "Training cfg  : $TRAINING_CONFIG"
echo ""

# ── Build metadata string ─────────────────────────────────────────────────────
METADATA="training-config=${TRAINING_CONFIG}"
if [ -n "$WANDB_API_KEY" ]; then
  METADATA="${METADATA},wandb-api-key=${WANDB_API_KEY}"
fi

# ── Create the VM ─────────────────────────────────────────────────────────────
echo "Creating $INSTANCE_NAME..."

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="pd-balanced" \
  --provisioning-model="SPOT" \
  --instance-termination-action="STOP" \
  --no-restart-on-failure \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --metadata="$METADATA" \
  --metadata-from-file="startup-script=${SCRIPT_DIR}/startup.sh"

echo ""
echo "VM created. Monitor startup (takes ~5 min):"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tail -f /tmp/rlgames-startup.log'"
echo ""
echo "Once setup is complete, SSH in and attach to the training session:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  screen -r training"
echo ""
echo "To delete when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
