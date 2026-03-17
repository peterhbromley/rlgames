#!/usr/bin/env bash
# create_vm.sh — provision a GCE Spot VM with a T4 GPU for RL training.
#
# Prerequisites (run once on your local machine):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   ./gce/create_vm.sh
#
# To delete the VM when done:
#   gcloud compute instances delete rlgames-trainer --zone=us-central1-a

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="${GCLOUD_PROJECT:-$(gcloud config get-value project)}"
ZONE="us-central1-a"          # T4s are widely available here; change if needed
INSTANCE_NAME="rlgames-trainer"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"        # ~20-30 GB for base image, rest for deps + checkpoints
# Deep Learning VM image: CUDA + PyTorch pre-installed
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

echo "Project : $PROJECT"
echo "Zone    : $ZONE"
echo "Instance: $INSTANCE_NAME"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type="pd-ssd" \
  --maintenance-policy="TERMINATE" \
  --provisioning-model="SPOT" \
  --instance-termination-action="STOP" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --metadata="install-nvidia-driver=True" \
  --metadata-from-file="startup-script=${SCRIPT_DIR}/startup.sh"

echo ""
echo "VM created. Connect with:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
