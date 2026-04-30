#!/usr/bin/env bash
# create_vm.sh — provision a GCE Spot VM for RL training.
#
# Prerequisites (run once on your local machine):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   # CPU-only
#   ./gce/create_vm.sh --zone us-central1-a --machine-type e2-highcpu-16
#
#   # GPU (T4 on N1)
#   ./gce/create_vm.sh --zone us-central1-a --machine-type n1-standard-4 --gpu nvidia-tesla-t4
#
#   # GPU (L4 on G2)
#   ./gce/create_vm.sh --zone us-central1-a --machine-type g2-standard-4 --gpu nvidia-l4
#
# Optional flags:
#   --name           — VM instance name (default: rlgames-trainer)
#   --gpu            — GPU type to attach (e.g. nvidia-tesla-t4, nvidia-l4)
#   --gpu-count      — Number of GPUs (default: 1)
#   --wandb-key      — WandB API key for experiment tracking
#   --project        — GCP project (default: gcloud config)
#
# GPU + machine-type compatibility:
#   nvidia-tesla-t4, nvidia-tesla-v100, nvidia-tesla-p100 → n1-* machines
#   nvidia-l4                                              → g2-* machines
#   nvidia-tesla-a100 / nvidia-a100-80gb                   → a2-* machines
#
# To delete the VM when done:
#   gcloud compute instances delete rlgames-trainer --zone=ZONE --quiet

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
ZONE=""
MACHINE_TYPE=""
INSTANCE_NAME="rlgames-trainer"
GPU_TYPE=""
GPU_COUNT="1"
WANDB_API_KEY=""
PROJECT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --zone)         ZONE="$2"; shift 2 ;;
    --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
    --name)         INSTANCE_NAME="$2"; shift 2 ;;
    --gpu)          GPU_TYPE="$2"; shift 2 ;;
    --gpu-count)    GPU_COUNT="$2"; shift 2 ;;
    --wandb-key)    WANDB_API_KEY="$2"; shift 2 ;;
    --project)      PROJECT="$2"; shift 2 ;;
    *)
      echo "Unknown flag: $1"
      echo "Usage: ./gce/create_vm.sh --zone ZONE --machine-type MACHINE_TYPE [--gpu GPU_TYPE] [--gpu-count N] [--name NAME] [--wandb-key KEY] [--project PROJECT]"
      exit 1
      ;;
  esac
done

if [ -z "$ZONE" ]; then
  echo "Error: --zone is required."
  echo "  Example: ./gce/create_vm.sh --zone us-central1-a --machine-type e2-highcpu-16"
  exit 1
fi

if [ -z "$MACHINE_TYPE" ]; then
  echo "Error: --machine-type is required."
  echo "  Example: ./gce/create_vm.sh --zone us-central1-a --machine-type e2-highcpu-16"
  exit 1
fi

PROJECT="${PROJECT:-$(gcloud config get-value project)}"

# ── Config ────────────────────────────────────────────────────────────────────
BOOT_DISK_SIZE="50GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

if [ -n "$GPU_TYPE" ]; then
  BOOT_DISK_SIZE="100GB"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Project       : $PROJECT"
echo "Zone          : $ZONE"
echo "Machine type  : $MACHINE_TYPE"
if [ -n "$GPU_TYPE" ]; then
echo "GPU           : ${GPU_COUNT}x ${GPU_TYPE}"
fi
echo "Instance      : $INSTANCE_NAME"
echo ""

# ── Build metadata ───────────────────────────────────────────────────────────
METADATA_ITEMS=()
if [ -n "$WANDB_API_KEY" ]; then
  METADATA_ITEMS+=("wandb-api-key=${WANDB_API_KEY}")
fi
if [ -n "$GPU_TYPE" ]; then
  METADATA_ITEMS+=("gpu=true")
fi

METADATA=""
if [ ${#METADATA_ITEMS[@]} -gt 0 ]; then
  METADATA=$(IFS=,; echo "${METADATA_ITEMS[*]}")
fi

# ── Create the VM ─────────────────────────────────────────────────────────────
echo "Creating $INSTANCE_NAME..."

CREATE_ARGS=(
  --project="$PROJECT"
  --zone="$ZONE"
  --machine-type="$MACHINE_TYPE"
  --image-family="$IMAGE_FAMILY"
  --image-project="$IMAGE_PROJECT"
  --boot-disk-size="$BOOT_DISK_SIZE"
  --boot-disk-type="pd-balanced"
  --provisioning-model="SPOT"
  --instance-termination-action="STOP"
  --no-restart-on-failure
  --scopes="https://www.googleapis.com/auth/cloud-platform"
  --metadata-from-file="startup-script=${SCRIPT_DIR}/startup.sh"
)

if [ -n "$GPU_TYPE" ]; then
  CREATE_ARGS+=(--accelerator="type=${GPU_TYPE},count=${GPU_COUNT}")
  CREATE_ARGS+=(--maintenance-policy="TERMINATE")
fi

if [ -n "$METADATA" ]; then
  CREATE_ARGS+=(--metadata="$METADATA")
fi

gcloud compute instances create "$INSTANCE_NAME" "${CREATE_ARGS[@]}"

echo ""
if [ -n "$GPU_TYPE" ]; then
  echo "VM created. Monitor startup (takes ~10 min with GPU driver install):"
else
  echo "VM created. Monitor startup (takes ~5 min):"
fi
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tail -f /tmp/rlgames-startup.log'"
echo ""
echo "Once setup is complete, SSH in to start training:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  cd /opt/rlgames/python"
echo "  uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml"
echo ""
echo "To delete when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
