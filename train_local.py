#!/usr/bin/env python3
"""
Local training script for testing without Modal.

Usage:
    python train_local.py --config configs/gpt_124m.yaml
    python train_local.py --config configs/test_local.yaml --no-wandb
"""

import argparse
import torch
import wandb
from pathlib import Path

from src.model.gpt import GPT, GPTConfig
from src.data.dataset import create_dataloader
from src.training.trainer import Trainer
from src.training.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train GPT locally")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cuda/mps/cpu)"
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Initialize W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_cfg = config.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "slm-from-scratch"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name", "local-experiment"),
            tags=wandb_cfg.get("tags", []) + ["local"],
            config=config,
        )

    # Create model
    print("Creating model...")
    model_cfg = config["model"]
    gpt_config = GPTConfig(**model_cfg)
    model = GPT(gpt_config)

    print(f"Model parameters: {model.count_parameters():,}")
    if use_wandb:
        wandb.log({"model/parameters": model.count_parameters()})

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
        use_wandb=use_wandb,
    )

    # Watch model with W&B
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    # Train
    print("Starting training...")
    print(f"Training for {train_cfg.get('max_steps', 50000)} steps")
    print(f"Checkpoints will be saved to: checkpoints/")

    trainer.train()

    if use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
