# Getting Started with SLM From Scratch

This guide will walk you through setting up your environment and running your first training experiment.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager (faster than pip)
- Modal account (for cloud GPU)
- Weights & Biases account (for experiment tracking)

## Step 1: Environment Setup

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create virtual environment and install dependencies

```bash
cd slm-from-scratch

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Step 2: Setup Modal

Modal provides serverless GPU compute. With your $30/month free credits, you can train 124M-350M models.

```bash
# Install Modal CLI
uv pip install modal

# Authenticate
modal setup
# This will open a browser for authentication

# Create data volume for datasets and checkpoints
modal volume create slm-data
```

## Step 3: Setup Weights & Biases

W&B provides free experiment tracking for individuals.

1. Create account at https://wandb.ai/signup
2. Get your API key from https://wandb.ai/authorize
3. Create Modal secret with your W&B API key:

```bash
modal secret create wandb WANDB_API_KEY=<your-api-key>
```

## Step 4: Test Local Setup

Before running on Modal, test everything works locally:

### Test the model

```bash
cd slm-from-scratch
python -c "from src.model.gpt import GPT, GPTConfig; \
    config = GPTConfig(n_layers=6, d_model=384); \
    model = GPT(config); \
    print(f'Model created with {model.count_parameters():,} parameters')"
```

Expected output:
```
Model created with 40,123,457 parameters
```

### Test the tokenizer

```bash
python -c "from src.data.tokenizer import get_tokenizer; \
    tokenizer = get_tokenizer(); \
    text = 'Hello, world!'; \
    tokens = tokenizer.encode(text); \
    print(f'Text: {text}'); \
    print(f'Tokens: {tokens}'); \
    print(f'Decoded: {tokenizer.decode(tokens)}')"
```

### Download test dataset

```bash
python scripts/download_data.py --dataset shakespeare --output data/
```

## Step 5: Run Your First Training (Local)

Let's do a quick local training run on TinyShakespeare to verify everything works:

```bash
# Create a minimal config for testing
cat > configs/test_local.yaml << 'EOF'
model:
  vocab_size: 50257
  context_length: 256
  n_layers: 6
  n_heads: 6
  d_model: 384
  d_ff: 1536
  dropout: 0.1

training:
  batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 3e-4
  warmup_steps: 100
  max_steps: 1000
  weight_decay: 0.1
  grad_clip: 1.0
  use_amp: true
  dtype: "bfloat16"
  checkpoint_every: 500
  eval_every: 250
  log_every: 50

data:
  dataset: "shakespeare"
  num_workers: 0

wandb:
  project: "slm-from-scratch"
  name: "test-local"
  tags: ["test", "local", "shakespeare"]
EOF

# Run local training (optional: --no-wandb to skip W&B)
python -c "
from src.model.gpt import GPT, GPTConfig
from src.data.dataset import create_dataloader
from src.training.trainer import Trainer
from src.training.config import load_config
import torch

config = load_config('configs/test_local.yaml')
gpt_config = GPTConfig(**config['model'])
model = GPT(gpt_config)

print(f'Model: {model.count_parameters():,} parameters')

# Create dataloader
train_loader = create_dataloader(
    dataset_name='shakespeare',
    batch_size=4,
    context_length=256,
    num_workers=0,
)

# Quick training test
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    config=config,
    device='cpu',  # Use CPU for quick test
    use_wandb=False,  # Disable W&B for test
)

print('Running 10 training steps...')
for i in range(10):
    batch = next(iter(train_loader))
    loss = trainer.train_step(batch)
    print(f'Step {i+1}, Loss: {loss:.4f}')

print('Local test successful!')
"
```

## Step 6: Run Training on Modal Cloud

Now that everything works locally, let's run real training on Modal GPUs!

### Upload dataset to Modal Volume

```bash
# TinyStories will auto-download, but you can pre-upload Shakespeare
modal volume put slm-data ./data/shakespeare.txt /shakespeare.txt
```

### Run training on T4 GPU (124M model)

```bash
# Run in detached mode (continues even if you disconnect)
modal run --detach modal_app.py::train --config configs/gpt_124m.yaml
```

This will:
- ✅ Spin up a T4 GPU instance (~$0.59/hour)
- ✅ Download TinyStories dataset
- ✅ Train 124M parameter GPT model
- ✅ Log everything to W&B
- ✅ Save checkpoints every 1000 steps

### Monitor training

1. **W&B Dashboard**: https://wandb.ai/<your-username>/slm-from-scratch
   - Real-time loss curves
   - GPU utilization
   - System metrics

2. **Modal Logs**:
   ```bash
   modal app logs slm-training
   ```

### Run training on A100 GPU (350M model)

For larger models, upgrade to A100:

```bash
modal run --detach modal_app.py::train --config configs/gpt_350m.yaml
```

Cost: ~$2.10/hour (A100-40GB)

## Step 7: Next Steps

Now that you have training working, you can:

1. **Follow Raschka's book** - Implement improvements chapter by chapter
2. **Experiment with hyperparameters** - Modify configs and compare in W&B
3. **Try different datasets** - Switch between Shakespeare and TinyStories
4. **Scale up gradually** - Start with 124M, then move to 350M
5. **Add features** - Implement FlashAttention, better sampling, etc.

## Common Issues

### "CUDA out of memory"
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use smaller GPU first (T4 instead of A100)

### "Modal authentication failed"
```bash
modal setup  # Re-authenticate
```

### "W&B not logging"
```bash
# Check secret is created
modal secret list

# Recreate secret
modal secret create wandb WANDB_API_KEY=<your-key>
```

### Slow training
- Use `dtype: "bfloat16"` for faster training
- Increase `num_workers` for data loading
- Use A100 instead of T4 for 2-3× speedup

## Budget Tips

With $30/month:
- Debug on T4 ($0.59/hr) = ~50 hours
- Short experiments on L4 ($0.80/hr) = ~37 hours
- Final training on A100-40GB ($2.10/hr) = ~14 hours

**Strategy**: Do most work on T4/L4, save A100 for final runs!

## Learning Resources

- 📖 [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- 💻 [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- 📚 [Modal Docs](https://modal.com/docs)
- 📊 [W&B Docs](https://docs.wandb.ai/)

---

**Ready to start learning? Run that first Modal training job! 🚀**
