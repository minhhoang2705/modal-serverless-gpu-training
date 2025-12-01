# Troubleshooting Guide

Common issues and solutions for the SLM From Scratch project.

---

## Installation Issues

### ❌ Issue: `uv pip install -e ".[dev]"` fails with hatchling error

**Error message:**
```
× Failed to build `slm-from-scratch`
├─▶ The build backend returned an error
╰─▶ Call to `hatchling.build.build_editable` failed
```

**Solution:** ✅ Already fixed in the current `pyproject.toml`

The issue was with the build backend. We switched from `hatchling` to `setuptools`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

**To fix manually** (if you encounter this):
```bash
# Edit pyproject.toml and change the build-system section
# Then reinstall
uv pip install -e ".[dev]"
```

---

## Environment Issues

### ❌ Issue: "No module named 'src'"

**Solution:**
Make sure you installed the package in editable mode:
```bash
uv pip install -e ".[dev]"
```

The `-e` flag creates a link to your source code, allowing imports from `src/`.

### ❌ Issue: Virtual environment not activated

**Check:**
```bash
which python
# Should show: .venv/bin/python
```

**Solution:**
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

---

## Import Issues

### ❌ Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Check if installed
uv pip list | grep torch

# If not installed
uv pip install torch
```

### ❌ Issue: "ModuleNotFoundError: No module named 'tiktoken'"

**Solution:**
```bash
uv pip install tiktoken
```

---

## Model Issues

### ❌ Issue: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** in your config:
```yaml
training:
  batch_size: 4  # Reduce from 8 or 16
  gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

2. **Use mixed precision:**
```yaml
training:
  use_amp: true
  dtype: "bfloat16"
```

3. **Use gradient checkpointing** (add to `src/model/gpt.py`):
```python
# In TransformerBlock.forward()
if self.training:
    x = checkpoint(lambda x: self.attention(self.ln1(x)), x)
```

4. **Use smaller model:**
```yaml
model:
  n_layers: 6   # Reduce from 12
  d_model: 384  # Reduce from 768
```

### ❌ Issue: Training is very slow

**Solutions:**

1. **Enable mixed precision:**
```yaml
training:
  use_amp: true
  dtype: "bfloat16"
```

2. **Increase data workers:**
```yaml
data:
  num_workers: 4  # Or higher
```

3. **Use faster GPU on Modal:**
```yaml
modal:
  gpu: "A100-40GB"  # Instead of T4
```

---

## Modal Issues

### ❌ Issue: "modal: command not found"

**Solution:**
```bash
uv pip install modal
modal setup
```

### ❌ Issue: Modal authentication failed

**Solution:**
```bash
# Re-authenticate
modal setup

# This will open a browser for authentication
```

### ❌ Issue: "Secret 'wandb' not found"

**Solution:**
```bash
# Get your W&B API key from https://wandb.ai/authorize
modal secret create wandb WANDB_API_KEY=<your-api-key>

# Verify
modal secret list
```

### ❌ Issue: Modal volume not found

**Solution:**
```bash
# Create the volume
modal volume create slm-data

# Verify
modal volume list
```

---

## W&B Issues

### ❌ Issue: W&B not logging

**Check:**
1. Is the secret created?
```bash
modal secret list | grep wandb
```

2. Is `use_wandb=True` in your code?

**Solution:**
```bash
# Recreate secret
modal secret create wandb WANDB_API_KEY=<your-key>

# Test locally
wandb login
```

### ❌ Issue: "wandb: ERROR Error uploading"

**Solution:**
This is often a network issue. W&B will retry automatically. If persistent:
```bash
# Check internet connection
# Check W&B status: https://status.wandb.ai/
```

---

## Training Issues

### ❌ Issue: Loss is NaN

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions:**

1. **Lower learning rate:**
```yaml
training:
  learning_rate: 1e-4  # Instead of 3e-4
```

2. **Stronger gradient clipping:**
```yaml
training:
  grad_clip: 0.5  # Instead of 1.0
```

3. **Longer warmup:**
```yaml
training:
  warmup_steps: 4000  # Instead of 2000
```

4. **Use mixed precision:**
```yaml
training:
  dtype: "bfloat16"  # More stable than float16
```

### ❌ Issue: Loss not decreasing

**Check:**

1. **Are gradients flowing?**
```python
# Add to training loop
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

2. **Is learning rate too small?**
```yaml
training:
  learning_rate: 5e-4  # Increase from 1e-4
```

3. **Is data loading correctly?**
```python
# Test data
batch = next(iter(train_loader))
print(batch[0].shape, batch[1].shape)
```

---

## Data Issues

### ❌ Issue: "Dataset not found"

**Solution:**
```bash
# For Shakespeare
python scripts/download_data.py --dataset shakespeare

# For TinyStories (auto-downloads, but can verify)
python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories', split='train[:10]')"
```

### ❌ Issue: Data loading is slow

**Solutions:**

1. **Increase workers:**
```yaml
data:
  num_workers: 8  # Increase from 4
```

2. **Pre-tokenize dataset:**
Save tokenized data to disk and load from there.

---

## MacOS Specific Issues

### ❌ Issue: "MPS backend out of memory"

On Apple Silicon (M1/M2/M3):

**Solution:**
```bash
# Fall back to CPU
python train_local.py --config configs/gpt_124m.yaml --device cpu
```

Or reduce batch size significantly:
```yaml
training:
  batch_size: 2
```

### ❌ Issue: Slow training on Mac

**Note:** MPS (Metal Performance Shaders) on Mac is slower than CUDA. For serious training, use Modal with GPU.

**Workaround for local testing:**
- Use very small models (n_layers=4, d_model=256)
- Use tiny datasets (Shakespeare)
- Keep max_steps low (500-1000)

---

## Quick Verification Tests

Run these to verify everything works:

### Test 1: Dependencies
```bash
python -c "import torch, wandb, tiktoken, datasets, modal; print('✅ All imports OK')"
```

### Test 2: Model
```bash
python -c "from src.model.gpt import GPT, GPTConfig; config = GPTConfig(n_layers=6, d_model=384); model = GPT(config); print(f'✅ Model: {model.count_parameters():,} params')"
```

### Test 3: Tokenizer
```bash
python -c "from src.data.tokenizer import get_tokenizer; tokenizer = get_tokenizer(); print(f'✅ Tokenizer: {tokenizer.n_vocab} vocab')"
```

### Test 4: Data Loading
```bash
python -c "from src.data.dataset import create_dataloader; loader = create_dataloader('shakespeare', batch_size=2, context_length=128, num_workers=0); batch = next(iter(loader)); print(f'✅ Data: {batch[0].shape}')"
```

---

## Still Having Issues?

### Check versions:
```bash
python --version  # Should be 3.10+
uv --version
modal --version
```

### Check environment:
```bash
which python  # Should point to .venv
pip list | grep torch
pip list | grep wandb
```

### Nuclear option (fresh start):
```bash
# Delete and recreate environment
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Getting Help

If you're still stuck:

1. **Check the error message carefully** - Often contains the solution
2. **Search the error** - Stack Overflow, GitHub Issues
3. **Check versions** - Make sure dependencies are compatible
4. **Try minimal reproduction** - Isolate the issue

---

**Most common fix:** Just reinstall! 🔄
```bash
uv pip install -e ".[dev]" --force-reinstall
```
