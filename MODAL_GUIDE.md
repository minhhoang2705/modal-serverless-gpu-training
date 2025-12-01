# Modal Cloud Training Guide

Complete guide for running your SLM training on Modal's cloud GPUs.

---

## Quick Start

### 1. Setup Modal (One-time)

```bash
# Install Modal
uv pip install modal

# Authenticate
modal setup
# This opens a browser for OAuth login

# Create W&B secret
modal secret create wandb WANDB_API_KEY=<your-wandb-api-key>

# Create data volume
modal volume create slm-data
```

### 2. Run Training

```bash
# Simple: Run with default config (124M model on T4)
modal run modal_app.py

# Recommended: Run in detached mode with 350M config
modal run --detach modal_app.py --config configs/gpt_350m.yaml

# Advanced: Direct function call
modal run modal_app.py::train --config-path configs/gpt_124m.yaml
```

---

## Modal Architecture

### How It Works

```
Your Computer                  Modal Cloud
     │                              │
     │  modal run modal_app.py      │
     ├──────────────────────────────>│
     │                              │
     │                         ┌────┴────┐
     │                         │ Build   │
     │                         │ Image   │
     │                         └────┬────┘
     │                              │
     │                         ┌────┴────────┐
     │                         │ 1. Copy src/│
     │                         │ 2. Copy cfg/│
     │                         │ 3. Install  │
     │                         └────┬────────┘
     │                              │
     │                         ┌────┴────────┐
     │                         │ Start GPU   │
     │                         │ (T4/A100)   │
     │                         └────┬────────┘
     │                              │
     │                         ┌────┴────────┐
     │                         │ Run train() │
     │                         │ function    │
     │                         └────┬────────┘
     │                              │
     │                         ┌────┴────────┐
     │    Monitor progress     │ W&B Logging │
     │<────────────────────────┤ Checkpoint  │
                               └─────────────┘
```

### Container Structure

When Modal runs your code, it creates a container with:

```
/root/
├── src/                    # Your source code
│   ├── model/
│   ├── data/
│   └── training/
├── configs/                # Your config files
│   ├── gpt_124m.yaml
│   └── gpt_350m.yaml
├── modal_app.py           # Entry point
└── /data/                 # Modal Volume (persistent storage)
    ├── datasets/
    ├── checkpoints/
    └── models/
```

---

## Configuration

### GPU Selection

Edit your config file to change GPU:

```yaml
# configs/gpt_350m.yaml
modal:
  gpu: "A100-40GB"  # Change to T4, L4, A10, A100-80GB, H100
  timeout: 43200    # 12 hours in seconds
```

Or override in command:
```bash
# Not currently supported - edit config file instead
```

### Available GPUs

| GPU | VRAM | Speed | Cost/hr | Best For |
|-----|------|-------|---------|----------|
| T4 | 16GB | 1× | $0.59 | Debugging, 124M |
| L4 | 24GB | 2× | $0.80 | 350M training |
| A10 | 24GB | 2.5× | $1.10 | Faster 350M |
| A100-40GB | 40GB | 5× | $2.10 | 774M models |
| A100-80GB | 80GB | 5× | $2.50 | 1B+ models |
| H100 | 80GB | 8× | $3.95 | Fastest |

---

## Monitoring

### W&B Dashboard

All metrics logged to: `https://wandb.ai/<your-username>/slm-from-scratch`

View:
- Real-time loss curves
- Learning rate schedule
- GPU utilization
- Generated text samples
- Checkpoints

### Modal Logs

```bash
# View logs in real-time
modal app logs slm-training --follow

# View past logs
modal app logs slm-training

# List all apps
modal app list
```

### Check Running Jobs

```bash
# List running functions
modal app show slm-training
```

---

## Data Management

### Upload Data to Volume

```bash
# Upload dataset
modal volume put slm-data ./local/data /data/datasets

# Upload checkpoints
modal volume put slm-data ./checkpoints /data/checkpoints
```

### Download from Volume

```bash
# Download trained model
modal volume get slm-data /data/models/gpt-350m_final.pt ./models/

# Download checkpoints
modal volume get slm-data /data/checkpoints ./checkpoints/
```

### List Volume Contents

```bash
modal volume ls slm-data /
modal volume ls slm-data /data/checkpoints
```

---

## Checkpointing

Training automatically saves checkpoints:

```
/data/checkpoints/
├── step_1000.pt
├── step_2000.pt
├── step_3000.pt
...
└── final.pt
```

Frequency controlled in config:
```yaml
training:
  checkpoint_every: 1000  # Save every N steps
```

Download checkpoints:
```bash
modal volume get slm-data /data/checkpoints ./checkpoints/
```

---

## Cost Management

### Estimate Costs

```python
# Example: 350M model on A100-40GB
hours = 12  # Expected training time
cost_per_hour = 2.10
total_cost = hours * cost_per_hour
print(f"Estimated cost: ${total_cost}")  # $25.20
```

### Budget Optimization

1. **Debug on T4** ($0.59/hr)
   - Test config changes
   - Verify code works
   - Run for 10-100 steps

2. **Short runs on L4** ($0.80/hr)
   - Hyperparameter search
   - Validation

3. **Final training on A100** ($2.10/hr)
   - Full 50k-100k steps
   - Best config only

### Track Spending

```bash
# View Modal usage
modal stats

# Check billing
# https://modal.com/settings/billing
```

---

## Troubleshooting

### Error: "No module named 'src'"

**Fix:** Already fixed in latest `modal_app.py`. Ensure you have:
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(...)
    .add_local_dir("src", remote_path="/root/src")      # This line
    .add_local_dir("configs", remote_path="/root/configs")  # This line
)
```

### Error: "Secret 'wandb' not found"

**Fix:**
```bash
# Create the secret
modal secret create wandb WANDB_API_KEY=<your-key>

# Verify
modal secret list
```

### Error: "Volume 'slm-data' not found"

**Fix:**
```bash
modal volume create slm-data
```

### Training Stops Unexpectedly

**Check:**
1. Timeout setting in config (default 12 hours)
2. Modal logs for errors
3. W&B for training progress

**Fix:**
```yaml
modal:
  timeout: 86400  # 24 hours
```

### Out of Memory

**Fix:**
```yaml
training:
  batch_size: 4        # Reduce from 8
  gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

Or use larger GPU:
```yaml
modal:
  gpu: "A100-80GB"  # Instead of A100-40GB
```

---

## Advanced Usage

### Resume from Checkpoint

Modify `modal_app.py` to load checkpoint:

```python
# In train() function, after model creation:
checkpoint_path = "/data/checkpoints/step_10000.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Resumed from {checkpoint_path}")
```

### Custom Training Script

Create `custom_modal_app.py`:
```python
import modal

app = modal.App("custom-training")

@app.function(
    image=modal.Image.debian_slim()
        .pip_install("torch", "wandb")
        .copy_local_dir("src", "/root/src"),
    gpu="A100",
    secrets=[modal.Secret.from_name("wandb")],
)
def custom_train():
    # Your custom training logic
    pass
```

Run:
```bash
modal run custom_modal_app.py::custom_train
```

### Multi-GPU Training

For very large models:
```python
@app.function(
    gpu="A100:4",  # 4× A100 GPUs
    # Implement DDP/FSDP in training code
)
```

---

## Best Practices

### 1. Test Locally First

```bash
# Always test config locally before Modal
python train_local.py --config configs/gpt_350m.yaml --no-wandb --device cpu
```

### 2. Start Small, Scale Up

```
T4 (debug) → L4 (validate) → A100 (final)
```

### 3. Use Detached Mode

```bash
modal run --detach modal_app.py --config configs/gpt_350m.yaml
```

Advantages:
- Can close terminal
- Won't lose progress if connection drops
- Logs saved in Modal

### 4. Monitor Actively

- Check W&B dashboard frequently
- Watch for:
  - Loss decreasing steadily
  - No NaN values
  - GPU utilization >90%

### 5. Checkpoint Frequently

```yaml
training:
  checkpoint_every: 500  # More frequent for expensive runs
```

---

## Common Workflows

### Workflow 1: Quick Experiment

```bash
# 1. Edit config
vim configs/gpt_124m.yaml

# 2. Test locally (5 min)
python train_local.py --config configs/gpt_124m.yaml --no-wandb

# 3. Run on Modal T4 (30 min)
modal run modal_app.py --config configs/gpt_124m.yaml

# 4. Check W&B
open https://wandb.ai/<username>/slm-from-scratch
```

### Workflow 2: Full Training

```bash
# 1. Create custom config
cp configs/gpt_350m.yaml configs/my_experiment.yaml
vim configs/my_experiment.yaml

# 2. Validate config
python -c "from src.training.config import load_config; load_config('configs/my_experiment.yaml')"

# 3. Launch training (detached)
modal run --detach modal_app.py --config configs/my_experiment.yaml

# 4. Monitor
modal app logs slm-training --follow

# 5. Download model when done
modal volume get slm-data /data/models/my_experiment_final.pt ./
```

### Workflow 3: Hyperparameter Search

```bash
# Create configs for different LRs
for lr in 1e-4 3e-4 1e-3; do
    sed "s/learning_rate: .*/learning_rate: $lr/" configs/gpt_124m.yaml > configs/gpt_124m_lr${lr}.yaml
done

# Run all in parallel (careful with budget!)
modal run --detach modal_app.py --config configs/gpt_124m_lr1e-4.yaml &
modal run --detach modal_app.py --config configs/gpt_124m_lr3e-4.yaml &
modal run --detach modal_app.py --config configs/gpt_124m_lr1e-3.yaml &

# Compare in W&B
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `modal setup` | Authenticate |
| `modal run modal_app.py` | Run training |
| `modal run --detach ...` | Background training |
| `modal app logs slm-training` | View logs |
| `modal app list` | List running apps |
| `modal volume ls slm-data` | List volume files |
| `modal volume get ...` | Download files |
| `modal secret list` | List secrets |
| `modal stats` | View usage stats |

---

**Ready to train on cloud GPUs! 🚀**

For more details: https://modal.com/docs
