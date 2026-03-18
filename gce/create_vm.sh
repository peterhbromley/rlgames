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
#   gcloud compute instances delete rlgames-trainer --zone=ZONE

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT="${GCLOUD_PROJECT:-$(gcloud config get-value project)}"
INSTANCE_NAME="rlgames-trainer"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"        # ~20-30 GB for base image, rest for deps + checkpoints
# Deep Learning VM image: CUDA + PyTorch pre-installed
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

# Zones to probe in order — US regions with good historical T4 availability.
CANDIDATE_ZONES=(
  us-east1-c
  us-east1-b
  us-east1-d
  us-west1-b
  us-west1-a
  us-east4-b
  us-east4-c
  us-central1-a
  us-central1-b
  us-central1-c
  us-central1-f
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Find an available zone ────────────────────────────────────────────────────
echo "Project : $PROJECT"
echo "Probing zones for T4 Spot availability..."
echo ""

ZONE=""
for candidate in "${CANDIDATE_ZONES[@]}"; do
  echo -n "  $candidate ... "
  result=$(gcloud compute instances create "${INSTANCE_NAME}-probe" \
    --project="$PROJECT" \
    --zone="$candidate" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --maintenance-policy="TERMINATE" \
    --provisioning-model="SPOT" \
    --instance-termination-action="DELETE" \
    --no-restart-on-failure \
    --quiet 2>&1)
  if echo "$result" | grep -q "Created"; then
    echo "available"
    gcloud compute instances delete "${INSTANCE_NAME}-probe" \
      --zone="$candidate" --quiet 2>/dev/null
    ZONE="$candidate"
    break
  else
    echo "unavailable"
  fi
done

if [ -z "$ZONE" ]; then
  echo ""
  echo "No T4 Spot capacity found in any candidate zone. Try again later."
  exit 1
fi

# ── Create the real VM ────────────────────────────────────────────────────────
echo ""
echo "Creating $INSTANCE_NAME in $ZONE..."

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
echo ""
echo "To delete when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
