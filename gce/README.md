# GCE GPU Training Setup

Step-by-step guide for training on a Google Compute Engine Spot VM with a T4 GPU.

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
chmod +x gce/create_vm.sh
./gce/create_vm.sh
```

**What this does:**
- Creates a `n1-standard-4` VM (4 vCPUs, 15 GB RAM) with a single T4 GPU in `us-central1-a`.
- Uses the `pytorch-latest-gpu` Deep Learning VM image, which comes with Ubuntu, CUDA, cuDNN, and PyTorch pre-installed — no manual CUDA setup needed.
- Sets `--provisioning-model=SPOT` for ~70% cost savings. Spot VMs can be preempted by Google, but `--instance-termination-action=STOP` means the disk is preserved on preemption (rather than deleted), so a resumed run can pick up from its last checkpoint.
- Sets `--maintenance-policy=TERMINATE` — required for any VM with a GPU, because GPU VMs cannot live-migrate.

**Expected cost:** ~$0.35–0.50/hour while running (T4 Spot in us-central1).

---

## Step 2 — Wait for startup to complete

The VM runs `gce/startup.sh` automatically on first boot. It clones the repo,
installs uv, installs all Python dependencies, and verifies the GPU is visible.
This takes **3–5 minutes** on first boot.

To watch progress after SSHing in:

```bash
gcloud compute ssh rlgames-trainer --zone=us-central1-a
tail -f /var/log/rlgames-startup.log
```

You'll see the startup script's output stream in real time. When you see
`=== rlgames startup complete ===` the VM is ready.

---

## Step 3 — Configure WandB

WandB tracks your training metrics (loss, reward) and saves them to the cloud so you can monitor progress from your local browser even after the SSH session closes.

```bash
uv run wandb login
# Paste your API key from https://wandb.ai/authorize
```

Your API key is stored in `~/.netrc` and persists across sessions on this VM.

---

## Step 4 — Run training

```bash
cd /opt/rlgames/python

# Full Oh Hell game, GPU, with WandB logging
uv run python -m training.train --config training/configs/oh_hell_full.yaml
```

Training logs to stdout and saves a checkpoint to `training/checkpoints/oh_hell_full.pt` every `log_interval` episodes (default: 10,000). If the VM is preempted and you restart, re-run the same command — it will automatically resume from the latest checkpoint.

To run in the background so training survives SSH disconnects:

```bash
nohup uv run python -m training.train --config training/configs/oh_hell_full.yaml \
  > logs/training.log 2>&1 &

echo $!  # prints the PID — save this if you want to stop the run later
```

Monitor progress after reconnecting:

```bash
tail -f logs/training.log
```

Check if it's still running:

```bash
ps aux | grep training.train
```

Stop it early:

```bash
kill <PID>
```

---

## Step 5 — Copy the checkpoint back

When training is done, download the checkpoint to your local machine:

```bash
# On your local machine
gcloud compute scp rlgames-trainer:/opt/rlgames/python/training/checkpoints/oh_hell_full.pt \
  ./python/training/checkpoints/ --zone=us-central1-a
```

---

## Step 6 — Stop the VM

**Always stop the VM when you are done.** You are charged for the GPU while it is running.

```bash
gcloud compute instances stop rlgames-trainer --zone=us-central1-a
```

The disk and its contents (checkpoints, logs) are preserved when stopped. You can restart later with:

```bash
gcloud compute instances start rlgames-trainer --zone=us-central1-a
```

To delete the VM entirely (frees all resources):

```bash
gcloud compute instances delete rlgames-trainer --zone=us-central1-a
```

---

## Troubleshooting

**`nvidia-smi` not found after SSH:**
The driver installation runs at first boot and can take a few minutes. Wait ~2 minutes and try again, or check `journalctl -u google-install-nvidia-driver`.

**VM preempted mid-training:**
Restart the VM (`gcloud compute instances start rlgames-trainer --zone=us-central1-a`), SSH back in, and re-run the training command. It will resume from the last checkpoint automatically.

**Permission denied running uv or writing files:**
The startup script runs as root, so `/opt/rlgames` may be owned by root. Fix with:
```bash
sudo chown -R $USER:$USER /opt/rlgames
```

**Spot VM unavailable in zone:**
Try another zone — T4 availability shifts constantly. See the zone-probing loop in the troubleshooting section above, or try `us-central1-b`, `us-east1-c`, `us-east1-d`, or `us-west1-b`.

**Out of GPU memory:**
Reduce `batch_size` in your config YAML (e.g. `128` instead of `256`).
