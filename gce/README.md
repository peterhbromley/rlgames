# GCE Training Setup

Step-by-step guide for training on a Google Compute Engine Spot VM.

---

## Prerequisites

Install and authenticate the `gcloud` CLI on your local machine. You only need to do this once.

```bash
# Install: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Make sure the Compute Engine API is enabled for your project:

```bash
gcloud services enable compute.googleapis.com
```

---

## Step 1 — Create the VM

```bash
# CPU-only
./gce/create_vm.sh --zone us-central1-a --machine-type e2-highcpu-16

# GPU (T4)
./gce/create_vm.sh --zone us-central1-a --machine-type n1-standard-4 --gpu nvidia-tesla-t4

# GPU (L4)
./gce/create_vm.sh --zone us-central1-a --machine-type g2-standard-4 --gpu nvidia-l4
```

**Required flags:**
- `--zone` — GCE zone (e.g. `us-central1-a`)
- `--machine-type` — GCE machine type (e.g. `e2-highcpu-16`, `n1-standard-4`)

**Optional flags:**
- `--name` — VM instance name (default: `rlgames-trainer`)
- `--gpu` — GPU type to attach (e.g. `nvidia-tesla-t4`, `nvidia-l4`)
- `--gpu-count` — Number of GPUs (default: 1)
- `--wandb-key` — WandB API key for experiment tracking
- `--project` — GCP project (default: current gcloud config)

**GPU + machine-type compatibility:**

| GPU | Machine type family |
|-----|-------------------|
| `nvidia-tesla-t4`, `nvidia-tesla-v100`, `nvidia-tesla-p100` | `n1-*` |
| `nvidia-l4` | `g2-*` |
| `nvidia-tesla-a100`, `nvidia-a100-80gb` | `a2-*` |

**What this does:**
- Creates a Spot VM with the specified machine type and zone.
- Uses Ubuntu 22.04 LTS.
- Sets `--provisioning-model=SPOT` for ~70% cost savings. Spot VMs can be preempted, but `--instance-termination-action=STOP` preserves the disk on preemption so you can resume from the last checkpoint.
- Runs `startup.sh` to install dependencies and clone the repo. **Does not start training** — you SSH in and start it manually.
- With `--gpu`: attaches the GPU, installs NVIDIA drivers, and installs PyTorch with CUDA support. Boot disk is increased to 100GB.

---

## Step 2 — Wait for startup to complete

The VM runs `gce/startup.sh` automatically on first boot. It clones the repo,
installs uv, and installs all Python dependencies.

To watch progress:

```bash
gcloud compute ssh rlgames-trainer --zone=us-central1-a -- 'tail -f /tmp/rlgames-startup.log'
```

**CPU VMs** complete in one boot (~5 min). The VM is ready when you see `=== rlgames startup complete ===`.

**GPU VMs** require two boots (~10 min total): the first boot installs everything including NVIDIA drivers and reboots automatically to load the kernel module. The VM is ready when you see `=== rlgames startup complete ===` after the second boot.

---

## Step 3 — Start training

SSH in and run whichever trainer you need:

```bash
gcloud compute ssh rlgames-trainer --zone=us-central1-a
cd /opt/rlgames/python
```

### Training commands

| Game | Algorithm | Command |
|------|-----------|---------|
| Oh Hell (4p, 3 tricks) | PPO self-play | `uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml` |
| Liars Dice (2p, 1 die) | Tabular MCCFR | `uv run python -m training.general.mccfr_trainer --config training/configs/liars_dice_mccfr.yaml` |
| Liars Dice (2p, 2 dice) | Deep CFR | `uv run python -m training.general.dcfr_trainer --config training/configs/liars_dice_dcfr.yaml` |

To run in the background so training survives SSH disconnects:

```bash
screen -S training
uv run python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml
# Ctrl-A D to detach; screen -r training to reattach
```

### WandB tracking

Each config has a `wandb` section. To enable, either edit the YAML (`enabled: true`) or pass `--wandb-key` when creating the VM.

Runs are organized by **group** within a single `rlgames` project:
- `oh-hell` — PPO self-play runs
- `liars-dice` — MCCFR and Deep CFR runs

In the WandB dashboard, use the **Group** dropdown to filter by game. Tags (`ppo`, `mccfr`, `deep-cfr`, `1-die`, `2-dice`, etc.) provide further filtering within a group.

---

## Step 4 — Copy the checkpoint back

When training is done, download the checkpoint to your local machine:

```bash
gcloud compute scp rlgames-trainer:/opt/rlgames/python/checkpoints/oh_hell_ppo.pt \
  ./python/checkpoints/ --zone=us-central1-a
```

---

## Step 5 — Stop the VM

**Always stop the VM when you are done.** You are charged while it is running.

```bash
gcloud compute instances stop rlgames-trainer --zone=us-central1-a
```

The disk is preserved when stopped. Restart later with:

```bash
gcloud compute instances start rlgames-trainer --zone=us-central1-a
```

To delete the VM entirely:

```bash
gcloud compute instances delete rlgames-trainer --zone=us-central1-a --quiet
```

---

## Troubleshooting

**VM preempted mid-training:**
Restart the VM, SSH back in, and re-run the training command. It will resume from the last checkpoint automatically.

**Permission denied running uv or writing files:**
```bash
sudo chown -R $USER:$USER /opt/rlgames
```

**Spot VM unavailable in zone:**
Try another zone — availability shifts. Try `us-central1-b`, `us-east1-c`, `us-west1-b`, etc.
