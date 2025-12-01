# Quick Reference Guide

## Common Commands

### Local Development

```bash
# Activate environment
source .venv/bin/activate

# Install/update dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Start Jupyter
jupyter notebook notebooks/
```

### Modal Commands

```bash
# Setup/auth
modal setup
modal token set --token-id xxx --token-secret yyy

# Volumes
modal volume create slm-data
modal volume ls slm-data /
modal volume put slm-data ./local/path /remote/path
modal volume get slm-data /remote/path ./local/path

# Secrets
modal secret create wandb WANDB_API_KEY=xxx
modal secret list

# Training
modal run modal_app.py::train --config configs/gpt_124m.yaml
modal run --detach modal_app.py::train --config configs/gpt_350m.yaml

# Logs
modal app logs slm-training
modal app logs slm-training --follow
```

### W&B Commands

```bash
# Login
wandb login

# View runs
wandb runs <project-name>

# Pull artifacts
wandb artifact get <run-id>/<artifact-name>
```

## GPU Options & Costs

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| T4 | 16GB | $0.59 | Debugging, 124M model |
| L4 | 24GB | $0.80 | 350M training |
| A10 | 24GB | $1.10 | Faster 350M |
| A100-40GB | 40GB | $2.10 | 774M+ models |
| A100-80GB | 80GB | $2.50 | 1B+ models |

## Model Sizes

| Config | Params | Layers | d_model | Training Time (TinyStories) |
|--------|--------|--------|---------|------------------------------|
| GPT-2 Small | 124M | 12 | 768 | ~4-6 hours (T4) |
| GPT-2 Medium | 350M | 24 | 1024 | ~10-12 hours (A100) |
| GPT-2 Large | 774M | 36 | 1280 | ~24+ hours (A100-80GB) |

## Config File Quickstart

```yaml
# configs/my_experiment.yaml
model:
  vocab_size: 50257
  context_length: 1024
  n_layers: 12
  n_heads: 12
  d_model: 768
  d_ff: 3072
  dropout: 0.1

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 3e-4
  warmup_steps: 2000
  max_steps: 50000
  use_amp: true
  dtype: "bfloat16"

data:
  dataset: "tinystories"

wandb:
  project: "slm-from-scratch"
  name: "my-experiment"
  tags: ["experiment"]

modal:
  gpu: "T4"
```

## Key Hyperparameters

| Param | Description | Typical Values |
|-------|-------------|----------------|
| `learning_rate` | Peak LR after warmup | 1e-4 to 1e-3 |
| `batch_size` | Per-device batch size | 4-32 |
| `gradient_accumulation_steps` | Effective batch = batch_size × this | 1-16 |
| `warmup_steps` | Linear LR warmup | 1-5% of max_steps |
| `max_steps` | Total training steps | 50k-200k |
| `weight_decay` | L2 regularization | 0.01-0.1 |
| `grad_clip` | Gradient clipping | 0.5-1.0 |

## Troubleshooting

### OOM (Out of Memory)
```yaml
# Reduce memory usage
training:
  batch_size: 4  # Smaller batch
  gradient_accumulation_steps: 8  # Maintain effective batch
  use_amp: true  # Mixed precision
  dtype: "bfloat16"
```

### Slow Training
```yaml
# Speed up training
training:
  dtype: "bfloat16"  # Faster than float16

data:
  num_workers: 4  # More data loading workers

modal:
  gpu: "A100-40GB"  # Faster GPU
```

### Training Unstable
```yaml
# Stabilize training
training:
  learning_rate: 1e-4  # Lower LR
  grad_clip: 0.5  # Stricter clipping
  warmup_steps: 4000  # Longer warmup
```

## Project Structure

```
slm-from-scratch/
├── src/
│   ├── model/gpt.py         # GPT architecture
│   ├── data/
│   │   ├── tokenizer.py     # BPE tokenization
│   │   └── dataset.py       # Data loading
│   └── training/
│       ├── trainer.py       # Training loop
│       └── config.py        # Config utilities
├── configs/                 # Training configs
├── modal_app.py             # Modal cloud deployment
├── scripts/                 # Utility scripts
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── README.md                # Overview
├── GETTING_STARTED.md       # Setup guide
└── QUICKREF.md              # This file
```

## Useful Code Snippets

### Test model locally
```python
from src.model.gpt import GPT, GPTConfig

config = GPTConfig(n_layers=6, d_model=384)
model = GPT(config)
print(f"Parameters: {model.count_parameters():,}")
```

### Generate text
```python
import torch
from src.model.gpt import GPT
from src.data.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
model = GPT.from_pretrained("checkpoints/final.pt")

prompt = "Once upon a time"
tokens = tokenizer.encode(prompt)
tokens = torch.tensor(tokens).unsqueeze(0)

generated = model.generate(tokens, max_new_tokens=50)
text = tokenizer.decode(generated[0].tolist())
print(text)
```

### Monitor W&B in terminal
```python
import wandb

api = wandb.Api()
runs = api.runs("username/slm-from-scratch")
for run in runs[:5]:
    print(f"{run.name}: loss={run.summary.get('train/loss'):.4f}")
```

## Week 1 Checklist

- [ ] Environment setup complete
- [ ] Modal authentication working
- [ ] W&B account created + secret added
- [ ] Local model test successful
- [ ] TinyShakespeare downloaded
- [ ] First Modal training job launched
- [ ] W&B dashboard showing metrics

## Next Steps

1. **Study Raschka's book** - Chapters 1-3 (tokenization, embeddings)
2. **Experiment locally** - Modify model architecture
3. **Run first real training** - 124M on TinyStories
4. **Analyze results** - Study W&B metrics
5. **Scale up** - Try 350M model

---

**Keep this file handy for quick reference! 📋**
