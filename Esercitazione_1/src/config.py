from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch import nn

@dataclass
class TrainingConfig:
    num_epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    loss_fn: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss())
    optimizer_type: str = "Adam"
    weight_decay: float = 0.0
    validation_split: float = 0.1
    num_workers: int = 2
    pin_memory: bool = True
    data_augmentation: List[str] = field(default_factory=lambda: ["RandomRotation", "RandomAffine"])
    output_dir: str = "./results"
    logging_steps: int = 100
    log_every_epoch: bool = True
    device: Optional[str] = None
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "DeepLearningApplication_Lab1"
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "accuracy"
    maximize_metric: bool = True
    
    # Telegram notifications
    use_telegram_notifications: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_additional_info: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
