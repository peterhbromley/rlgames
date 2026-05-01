#!/usr/bin/env bash
# create_vm.sh — provision a GCE Spot VM for RL training.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   # CPU
#   ./gce/create_vm.sh --zone us-central1-a --machine-type e2-highcpu-16
#
#   # GPU (T4 on N1, L4 on G2)
#   ./gce/create_vm.sh --zone us-west3-b --machine-type n1-standard-4 --gpu nvidia-tesla-t4
#   ./gce/create_vm.sh --zone us-west3-b --machine-type g2-standard-4 --gpu nvidia-l4
#
# GPU VMs use Google's Deep Learning VM image (NVIDIA drivers + CUDA pre-installed).

set -euo pipefail

ZONE=""
MACHINE_TYPE=""
INSTANCE_NAME="rlgames-trainer"
GPU_TYPE=""
GPU_COUNT="1"
WANDB_API_KEY=""
PROJECT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --zone)          ZONE="$2";          shift 2 ;;
    --machine-type)  MACHINE_TYPE="$2";  shift 2 ;;
    --name)          INSTANCE_NAME="$2"; shift 2 ;;
    --gpu)           GPU_TYPE="$2";      shift 2 ;;
    --gpu-count)     GPU_COUNT="$2";     shift 2 ;;
    --wandb-key)     WANDB_API_KEY="$2"; shift 2 ;;
    --project)       PROJECT="$2";       shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

[[ -z "$ZONE" ]]         && echo "Error: --zone is required"         && exit 1
[[ -z "$MACHINE_TYPE" ]] && echo "Error: --machine-type is required" && exit 1

PROJECT="${PROJECT:-$(gcloud config get-value project)}"

# GPU VMs use Deep Learning VM image (CUDA + drivers pre-installed, no setup needed).
# CPU VMs use standard Ubuntu 22.04.
if [[ -n "$GPU_TYPE" ]]; then
  IMAGE_FAMILY="common-cu124-debian-11"
  IMAGE_PROJECT="deeplearning-platform-release"
  DISK_SIZE="100GB"
else
  IMAGE_FAMILY="ubuntu-2204-lts"
  IMAGE_PROJECT="ubuntu-os-cloud"
  DISK_SIZE="50GB"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Project      : $PROJECT"
echo "Zone         : $ZONE"
echo "Machine type : $MACHINE_TYPE"
echo "Image        : $IMAGE_PROJECT/$IMAGE_FAMILY"
[[ -n "$GPU_TYPE" ]] && echo "GPU          : ${GPU_COUNT}x ${GPU_TYPE}"
echo "Instance     : $INSTANCE_NAME"
echo ""

METADATA=""
[[ -n "$WANDB_API_KEY" ]] && METADATA="wandb-api-key=${WANDB_API_KEY}"
[[ -n "$GPU_TYPE" ]]      && METADATA="${METADATA:+$METADATA,}gpu=true"

CREATE_ARGS=(
  --project="$PROJECT"
  --zone="$ZONE"
  --machine-type="$MACHINE_TYPE"
  --image-family="$IMAGE_FAMILY"
  --image-project="$IMAGE_PROJECT"
  --boot-disk-size="$DISK_SIZE"
  --boot-disk-type="pd-balanced"
  --provisioning-model="SPOT"
  --instance-termination-action="STOP"
  --no-restart-on-failure
  --scopes="https://www.googleapis.com/auth/cloud-platform"
  --metadata-from-file="startup-script=${SCRIPT_DIR}/startup.sh"
)

[[ -n "$GPU_TYPE" ]] && CREATE_ARGS+=(
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}"
  --maintenance-policy="TERMINATE"
)

[[ -n "$METADATA" ]] && CREATE_ARGS+=(--metadata="$METADATA")

echo "Creating $INSTANCE_NAME..."
gcloud compute instances create "$INSTANCE_NAME" "${CREATE_ARGS[@]}"

echo ""
echo "Monitor startup (~5 min):"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tail -f /tmp/rlgames-startup.log'"
echo ""
echo "Once you see '=== startup complete ===', SSH in:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  cd /opt/rlgames/python"
echo ""
echo "Delete when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
