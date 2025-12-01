"""
Modal app for training GPT models in the cloud.

Usage:
    # Run with default config (124M model)
    modal run modal_app.py

    # Run with specific config
    modal run modal_app.py --config configs/gpt_350m.yaml

    # Run in detached mode (recommended for long training)
    modal run --detach modal_app.py --config configs/gpt_350m.yaml

    # Direct function call (advanced)
    modal run modal_app.py::train --config-path configs/gpt_124m.yaml
"""

import modal

# Define Modal app
app = modal.App("slm-training")

# Create persistent volume for data and checkpoints
volume = modal.Volume.from_name("slm-data", create_if_missing=True)

# Define container image with dependencies
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
    # Copy source code into the container
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)


@app.function(
    image=image,
    gpu="H100",  # Default GPU, override with config
    timeout=60 * 60 * 12,  # 12 hour max
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config_path: str = "configs/gpt_124m.yaml"):
    """
    Train GPT model on Modal cloud.

    Args:
        config_path: Path to YAML config file (relative to /root in container)
    """
    import os
    import sys
    import torch
    import wandb
    from pathlib import Path

    # Change to /root directory where our code is
    os.chdir("/root")

    # Add /root to Python path for imports
    sys.path.insert(0, "/root")

    from src.model.gpt import GPT, GPTConfig
    from src.data.dataset import create_dataloader
    from src.training.trainer import Trainer
    from src.training.config import load_config

    # Load config
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)

    # Initialize W&B
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "slm-from-scratch"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", "experiment"),
        tags=wandb_cfg.get("tags", []),
        config=config,
    )

    # Create model
    print("Creating model...")
    model_cfg = config["model"]
    gpt_config = GPTConfig(**model_cfg)
    model = GPT(gpt_config)

    print(f"Model parameters: {model.count_parameters():,}")
    wandb.log({"model/parameters": model.count_parameters()})

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"GPU available: {torch.cuda.is_available()}")

    # Create dataloaders
    print("Creating dataloaders...")
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    train_loader = create_dataloader(
        dataset_name=data_cfg.get("dataset", "tinystories"),
        split=data_cfg.get("train_split", "train"),
        batch_size=train_cfg.get("batch_size", 8),
        context_length=model_cfg.get("context_length", 1024),
        num_workers=data_cfg.get("num_workers", 4),
    )

    val_loader = None
    if data_cfg.get("val_split"):
        val_loader = create_dataloader(
            dataset_name=data_cfg.get("dataset", "tinystories"),
            split=data_cfg.get("val_split", "validation"),
            batch_size=train_cfg.get("batch_size", 8) * 2,
            context_length=model_cfg.get("context_length", 1024),
            num_workers=data_cfg.get("num_workers", 4),
        )

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=True,
    )

    # Watch model with W&B
    wandb.watch(model, log="all", log_freq=100)

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model to volume
    final_path = f"/data/models/{wandb_cfg.get('name', 'model')}_final.pt"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)

    wandb.finish()
    print(f"Training complete! Model saved to {final_path}")


@app.local_entrypoint()
def main(config: str = "configs/gpt_124m.yaml"):
    """
    Local entrypoint for running training.

    Usage:
        modal run modal_app.py --config configs/gpt_124m.yaml
    """
    train.remote(config)
