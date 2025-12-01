"""Configuration loading utilities."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    max_steps: int = 50000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    use_amp: bool = True
    dtype: str = "bfloat16"

    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 10


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset: str = "tinystories"
    train_split: str = "train"
    val_split: str = "validation"
    num_workers: int = 4


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    project: str = "slm-from-scratch"
    entity: Optional[str] = None
    name: str = "experiment"
    tags: list = field(default_factory=list)


@dataclass
class ModalConfig:
    """Modal cloud configuration."""

    gpu: str = "A100"
    timeout: int = 7200


def load_config(config_path: str):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary with all sections
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # Test config loading
    config = load_config("configs/gpt_124m.yaml")
    print("Model config:", config["model"])
    print("Training config:", config["training"])
