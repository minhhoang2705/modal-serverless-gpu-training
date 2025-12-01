<div align="center">
  <h1>🚀 SLM From Scratch</h1>
  <p><strong>Build a GPT from scratch with cloud GPU training on Modal</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
  [![Modal](https://img.shields.io/badge/Modal-cloud_GPU-purple.svg)](https://modal.com)
  [![W&B](https://img.shields.io/badge/W%26B-experiment_tracking-orange.svg)](https://wandb.ai)
</div>

---

## 📖 Overview

A complete implementation of GPT (Generative Pre-trained Transformer) from scratch, designed for researchers who want to understand transformer architecture while learning serverless GPU training with **Modal**.

**What makes this different:**
- ☁️ **Cloud-first design** - No local GPU needed, train on Modal's serverless infrastructure
- 💰 **Cost-transparent** - Detailed budget breakdowns ($30 free credits covers full training)
- 📚 **Educational focus** - Every component documented for learning
- 🛠️ **Production-ready** - Type-safe configs, proper error handling, W&B monitoring

---

## 🏗️ Codebase Architecture

```
slm-from-scratch/
├── src/
│   ├── model/
│   │   ├── gpt.py              # GPT architecture (350 lines)
│   │   │   ├── GPTConfig       # Model configuration
│   │   │   ├── MultiHeadAttention
│   │   │   ├── FeedForward
│   │   │   ├── TransformerBlock
│   │   │   └── GPT             # Main model class
│   │   └── __init__.py
│   │
│   ├── data/
│   │   ├── tokenizer.py        # GPT-2 BPE tokenizer
│   │   ├── dataset.py          # PyTorch Dataset & DataLoader
│   │   └── __init__.py
│   │
│   └── training/
│       ├── trainer.py          # Training loop with W&B
│       ├── config.py           # Config dataclasses
│       └── __init__.py
│
├── configs/
│   ├── gpt_124m.yaml          # 124M parameter config (12 layers, 768 dim)
│   └── gpt_350m.yaml          # 350M parameter config (24 layers, 1024 dim)
│
├── modal_app.py               # 🎯 Modal deployment entrypoint
├── pyproject.toml             # Dependencies (uv package manager)
└── README.md
```

### Core Components

#### 1. **Model** (`src/model/gpt.py`)

Full GPT implementation with:
- **Multi-head self-attention** with causal masking
- **Pre-norm transformer blocks** (LayerNorm before attention/FFN)
- **Weight tying** (share embeddings with output projection)
- **Positional embeddings** (learned, not sinusoidal)

```python
from src.model.gpt import GPT, GPTConfig

# Create a 124M parameter model
config = GPTConfig(
    vocab_size=50257,      # GPT-2 tokenizer size
    context_length=1024,   # Max sequence length
    n_layers=12,           # Transformer blocks
    n_heads=12,            # Attention heads
    d_model=768,           # Hidden dimension
    d_ff=3072,             # FFN dimension (4 * d_model)
    dropout=0.1
)

model = GPT(config)
print(f"Parameters: {model.count_parameters():,}")  # 124,439,808
```

#### 2. **Data Pipeline** (`src/data/`)

Handles tokenization and data loading:

```python
from src.data.tokenizer import get_tokenizer
from src.data.dataset import create_dataloader

# Get GPT-2 tokenizer
tokenizer = get_tokenizer()

# Create DataLoader (handles HuggingFace datasets automatically)
train_loader = create_dataloader(
    dataset_name="roneneldan/TinyStories",  # HuggingFace dataset
    split="train",
    batch_size=16,
    context_length=1024,
    num_workers=4
)

# Or use custom texts
from src.data.dataset import TextDataset
texts = ["Your custom text here...", "More text..."]
dataset = TextDataset(texts, tokenizer, context_length=1024)
```

#### 3. **Training** (`src/training/trainer.py`)

Complete training loop with:
- Mixed precision (FP16/BF16) via PyTorch AMP
- Gradient accumulation
- Learning rate warmup + cosine decay
- Checkpointing
- W&B logging

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config_dict,     # From YAML
    device="cuda",
    use_wandb=True
)

trainer.train()  # Starts training loop
```

---

## ☁️ Modal Cloud GPU Deployment

### Why Modal?

Modal is a serverless platform that lets you run Python functions on cloud GPUs **without managing infrastructure**:

- 💸 **Pay-per-second billing** - Only pay when training
- 🎁 **$30/month free credits** - Enough for full project
- 🚀 **Zero setup** - No Docker, Kubernetes, or cloud config
- 📦 **Automatic dependency management** - Specify packages in code

### Setup (5 minutes)

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate (opens browser)
modal setup

# 3. Create W&B secret (for experiment tracking)
# Get API key from https://wandb.ai/settings
modal secret create wandb WANDB_API_KEY=your_key_here
```

### Modal App Explained (`modal_app.py`)

```python
import modal

# 1. Create Modal app
app = modal.App("slm-training")

# 2. Define persistent storage for datasets/checkpoints
volume = modal.Volume.from_name("slm-data", create_if_missing=True)

# 3. Build container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "tiktoken>=0.5.0",
        "datasets>=2.14.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
    )
    # Copy your source code into container
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

# 4. Define GPU training function
@app.function(
    image=image,
    gpu="A100",                              # GPU tier (T4/L4/A100/H100)
    timeout=60 * 60 * 12,                   # 12 hour max
    volumes={"/data": volume},               # Mount persistent storage
    secrets=[modal.Secret.from_name("wandb")]  # W&B API key
)
def train(config_path: str = "configs/gpt_124m.yaml"):
    import os, sys
    os.chdir("/root")
    sys.path.insert(0, "/root")

    from src.model.gpt import GPT, GPTConfig
    from src.data.dataset import create_dataloader
    from src.training.trainer import Trainer
    from src.training.config import load_config
    import wandb

    # Load config
    config = load_config(config_path)

    # Initialize W&B
    wandb.init(
        project="slm-from-scratch",
        config=config
    )

    # Create model
    model = GPT(GPTConfig(**config["model"]))

    # Create dataloaders
    train_loader = create_dataloader(
        dataset_name="roneneldan/TinyStories",
        split="train",
        batch_size=config["training"]["batch_size"],
        context_length=config["model"]["context_length"],
    )

    # Train
    trainer = Trainer(model, train_loader, config=config)
    trainer.train()

    wandb.finish()

# 5. Local entrypoint
@app.local_entrypoint()
def main(config: str = "configs/gpt_124m.yaml"):
    train.remote(config)  # .remote() runs on cloud GPU
```

---

## 🚀 Usage Examples

### Example 1: Quick Test (124M model, 2 hours, ~$1)

```bash
# Train 124M parameter model on T4 GPU
modal run modal_app.py --config configs/gpt_124m.yaml
```

**What happens:**
1. Modal builds container with dependencies (~2 min first time, cached after)
2. Spins up T4 GPU instance (~30 seconds)
3. Downloads TinyStories dataset from HuggingFace
4. Trains for 50,000 steps with W&B logging
5. Saves checkpoints every 1,000 steps to Modal Volume
6. GPU shuts down automatically when done

**Monitor training:** https://wandb.ai/your-username/slm-from-scratch

### Example 2: Production Training (350M model, detached)

```bash
# Run in background (detached mode)
modal run --detach modal_app.py --config configs/gpt_350m.yaml

# Check status
modal app logs slm-training

# Stop early if needed
modal app stop slm-training
```

### Example 3: Custom Configuration

Create `configs/custom.yaml`:

```yaml
model:
  vocab_size: 50257
  context_length: 2048      # Longer context
  n_layers: 16
  n_heads: 16
  d_model: 1024
  d_ff: 4096
  dropout: 0.1

training:
  batch_size: 8             # Smaller batch for larger model
  gradient_accumulation_steps: 4  # Effective batch = 32
  learning_rate: 1e-4
  max_steps: 100000
  warmup_steps: 5000

  checkpoint_every: 2000
  eval_every: 1000

data:
  dataset: "roneneldan/TinyStories"
  train_split: "train"
  val_split: "validation"

wandb:
  project: "slm-from-scratch"
  name: "custom-1B-run"
  tags: ["custom", "1B"]
```

Run it:
```bash
modal run modal_app.py --config configs/custom.yaml
```

### Example 4: Using Different Datasets

```python
# In modal_app.py, change dataset:

# TinyStories (2M stories, ~500MB)
dataset_name = "roneneldan/TinyStories"

# OpenWebText (38GB, Reddit links)
dataset_name = "Skylion007/openwebtext"

# Wikipedia (20GB)
dataset_name = "wikipedia"

# Or load from Modal Volume:
train_loader = create_dataloader(
    dataset_path="/data/my-custom-data.txt",  # Uploaded to volume
    split="train",
    ...
)
```

### Example 5: GPU Tier Strategy

```python
# configs/gpt_124m.yaml
modal:
  gpu: "T4"      # $0.50/hr - Development/testing
  timeout: 7200  # 2 hours

# configs/gpt_350m.yaml
modal:
  gpu: "L4"      # $0.70/hr - Production 350M
  timeout: 43200 # 12 hours

# configs/gpt_774m.yaml
modal:
  gpu: "A100"    # $0.90/hr - Large models
  timeout: 86400 # 24 hours
```

Update `modal_app.py`:
```python
@app.function(
    gpu=config.get("modal", {}).get("gpu", "A100"),  # Read from config
    timeout=config.get("modal", {}).get("timeout", 43200),
    ...
)
```

---

## 💰 Cost Breakdown

| Model | GPU | Time | Cost | Use Case |
|-------|-----|------|------|----------|
| 124M | T4 | 2h | $1 | Testing pipeline |
| 350M | L4 | 12h | $8 | Main training run |
| 774M | A100 | 24h | $22 | Advanced experiments |

**Budget strategy with $30 free credits:**
1. Test on 124M (T4) - $1
2. Production 350M run (L4) - $8
3. Hyperparameter tuning 3× (L4) - $24
4. **Total: $33** (pay $3 or optimize)

---

## 🔧 Configuration Reference

### Model Config (`configs/gpt_*.yaml`)

```yaml
model:
  vocab_size: 50257        # Vocabulary size (GPT-2 tokenizer)
  context_length: 1024     # Max sequence length
  n_layers: 12             # Number of transformer blocks
  n_heads: 12              # Attention heads (must divide d_model)
  d_model: 768             # Hidden dimension
  d_ff: 3072               # FFN dimension (typically 4 * d_model)
  dropout: 0.1             # Dropout rate
```

**Parameter count formula:**
- Embeddings: `vocab_size * d_model * 2` (token + position)
- Per layer: `4 * d_model^2 + 2 * d_model * d_ff`
- Total ≈ `12 * n_layers * d_model^2`

### Training Config

```yaml
training:
  batch_size: 16                    # Per-GPU batch size
  gradient_accumulation_steps: 2    # Effective batch = 32
  learning_rate: 3e-4               # Peak LR (AdamW)
  warmup_steps: 2000                # Linear warmup steps
  max_steps: 50000                  # Total training steps
  weight_decay: 0.1                 # L2 regularization
  grad_clip: 1.0                    # Gradient clipping

  # Optimizer (AdamW)
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1e-8

  # Mixed precision
  use_amp: true                     # Enable FP16/BF16
  dtype: "bfloat16"                 # Preferred over float16

  # Logging
  checkpoint_every: 1000            # Save checkpoint frequency
  eval_every: 500                   # Validation frequency
  log_every: 10                     # W&B logging frequency
```

---

## 📊 Monitoring with W&B

Training metrics automatically logged:
- `train/loss` - Training loss (every 10 steps)
- `train/lr` - Learning rate (cosine schedule)
- `train/tokens_seen` - Total tokens processed
- `val/loss` - Validation loss (every 500 steps)
- `model/parameters` - Parameter count
- Gradient histograms (every 100 steps)

**Dashboard:** https://wandb.ai/your-username/slm-from-scratch

---

## 🎯 Quick Start (Zero to Trained Model in 30 Minutes)

```bash
# 1. Clone and install (2 min)
git clone https://github.com/yourusername/slm-from-scratch
cd slm-from-scratch
pip install modal wandb

# 2. Setup Modal + W&B (3 min)
modal setup
modal secret create wandb WANDB_API_KEY=your_key

# 3. Start training (25 min for 124M on T4)
modal run modal_app.py --config configs/gpt_124m.yaml

# 4. Monitor
# Visit https://wandb.ai and watch training live!
```

---

## 🐛 Troubleshooting

### Error: "No module named 'src'"
**Solution:** Ensure `modal_app.py` has:
```python
.add_local_dir("src", remote_path="/root/src")
os.chdir("/root")
sys.path.insert(0, "/root")
```

### Error: "TypeError: '<=' not supported between float and str"
**Solution:** Config values need type conversion:
```python
learning_rate = float(config["training"]["learning_rate"])
max_steps = int(config["training"]["max_steps"])
```

### W&B not logging
**Solution:** Check Modal secret:
```bash
modal secret list  # Verify "wandb" exists
modal secret create wandb WANDB_API_KEY=your_key  # Recreate
```

### Out of memory (OOM)
**Solution:** Reduce batch size or use gradient accumulation:
```yaml
training:
  batch_size: 8              # Half the batch size
  gradient_accumulation_steps: 4  # Double accumulation
```

---

## 📚 Learning Resources

- **Book:** [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka
- **Code Reference:** [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- **Modal Docs:** [modal.com/docs](https://modal.com/docs)
- **W&B Docs:** [docs.wandb.ai](https://docs.wandb.ai/)
- **GPT-2 Paper:** [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 🤝 Contributing

This is an educational project - contributions welcome!

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📖 Citation

If you use this project in your research:

```bibtex
@software{slm_from_scratch,
  author = {Hoang-Minh Tran},
  title = {SLM From Scratch: Building GPT with Cloud GPU Training},
  year = {2025},
  url = {https://github.com/minhtranh/slm-from-scratch},
  note = {Educational implementation of GPT with Modal serverless training}
}
```

---

<div align="center">
  <p>Built with ❤️ for ML researchers learning transformers and cloud GPU training</p>
  <p><strong>Questions?</strong> Open an issue or discussion!</p>
</div>
