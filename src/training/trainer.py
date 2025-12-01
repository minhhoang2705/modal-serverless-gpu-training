"""Training loop with W&B integration."""

import math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from pathlib import Path


class Trainer:
    """
    Trainer class for language model training with W&B integration.

    Handles:
    - Training loop with gradient accumulation
    - Mixed precision training (AMP)
    - Checkpointing
    - W&B logging
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        config=None,
        device="cuda",
        use_wandb=True,
    ):
        """
        Args:
            model: The GPT model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration dict
            device: Device to train on ("cuda" or "cpu")
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        self.use_wandb = use_wandb

        # Training config
        train_cfg = self.config.get("training", {})
        self.learning_rate = float(train_cfg.get("learning_rate", 3e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.1))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.max_steps = int(train_cfg.get("max_steps", 50000))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 2000))

        self.checkpoint_every = int(train_cfg.get("checkpoint_every", 1000))
        self.eval_every = int(train_cfg.get("eval_every", 500))
        self.log_every = int(train_cfg.get("log_every", 10))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(float(train_cfg.get("adam_beta1", 0.9)), float(train_cfg.get("adam_beta2", 0.95))),
            eps=float(train_cfg.get("adam_eps", 1e-8)),
            weight_decay=self.weight_decay,
        )

        # Mixed precision
        use_amp_cfg = train_cfg.get("use_amp", True)
        # Convert string "true"/"false" to boolean if needed
        if isinstance(use_amp_cfg, str):
            self.use_amp = use_amp_cfg.lower() in ('true', '1', 'yes')
        else:
            self.use_amp = bool(use_amp_cfg)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler("cuda")
        else:
            self.scaler = None
            self.use_amp = False  # Disable AMP if no CUDA

        # Tracking
        self.global_step = 0
        self.tokens_seen = 0

    def get_lr(self):
        """Get current learning rate (with warmup)."""
        if self.global_step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * self.global_step / self.warmup_steps
        else:
            # Cosine decay after warmup
            progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1.0 + math.cos(progress * math.pi))

    def update_lr(self):
        """Update learning rate in optimizer."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: (x, y) tuple of input and target tensors

        Returns:
            loss: scalar loss value
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.use_amp):
            logits, loss = self.model(x, y)

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on validation set."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.use_amp):
                logits, loss = self.model(x, y)

            total_loss += loss.item()
            num_batches += 1

            # Only evaluate on a subset to save time
            if num_batches >= 50:
                break

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def train(self):
        """Main training loop."""
        self.model.train()
        self.optimizer.zero_grad()

        train_iterator = iter(self.train_loader)
        accumulated_loss = 0

        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Update learning rate
            self.update_lr()

            # Training step
            loss = self.train_step(batch)
            accumulated_loss += loss

            # Update tokens seen
            batch_size, seq_len = batch[0].shape
            self.tokens_seen += batch_size * seq_len

            # Optimizer step (every N accumulation steps)
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Log to W&B
                if self.global_step % self.log_every == 0:
                    metrics = {
                        "train/loss": accumulated_loss / self.gradient_accumulation_steps,
                        "train/lr": self.get_lr(),
                        "train/step": self.global_step,
                        "train/tokens_seen": self.tokens_seen,
                    }

                    if self.use_wandb:
                        wandb.log(metrics, step=self.global_step)

                    pbar.set_postfix({"loss": f"{metrics['train/loss']:.4f}"})

                accumulated_loss = 0

            # Evaluation
            if self.global_step % self.eval_every == 0 and self.val_loader is not None:
                val_loss = self.evaluate()
                if val_loss is not None:
                    if self.use_wandb:
                        wandb.log({"val/loss": val_loss}, step=self.global_step)
                    print(f"Validation loss: {val_loss:.4f}")

            # Checkpointing
            if self.global_step % self.checkpoint_every == 0:
                self.save_checkpoint(f"checkpoints/step_{self.global_step}.pt")

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        print("Training complete!")

        # Save final checkpoint
        self.save_checkpoint("checkpoints/final.pt")


if __name__ == "__main__":
    print("Trainer module - use train.py to run training")
