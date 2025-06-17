import torch
import numpy as np
import os
import copy
import time
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import wandb
from typing import Optional, Callable, Any, Dict
from .data import get_data_transforms, TransformedSubset
from sklearn.metrics import accuracy_score
from .telegram_notifier import TelegramNotifier, create_telegram_notifier_from_env

class StreamlinedTrainer:
    def __init__(self, model, config, train_dataset, test_dataset, compute_metrics_fn: Optional[Callable] = None):
        self.model = model
        self.config = config
        self.compute_metrics_fn = compute_metrics_fn or self._default_metrics
        self._set_seed(config.seed)
        self.train_loader, self.val_loader = self._setup_data_loaders(train_dataset)
        self.test_loader = self._setup_test_loader(test_dataset)
        self.model.to(config.device)
        self.optimizer = self._setup_optimizer()
        self.global_step = 0
        self.best_metric = None
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        # Initialize Telegram notifier
        self.telegram_notifier = None
        if config.use_telegram_notifications:
            if config.telegram_bot_token and config.telegram_chat_id:
                self.telegram_notifier = TelegramNotifier(
                    bot_token=config.telegram_bot_token,
                    chat_id=config.telegram_chat_id,
                    enabled=True
                )
            else:
                # Try to create notifier from environment variables
                self.telegram_notifier = create_telegram_notifier_from_env()
        
        # Timer for training duration
        self.training_start_time = None
        
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=str(model),
                config=config.__dict__
            )
            wandb.watch(model, log="all", log_freq=100)
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    def _setup_data_loaders(self, train_dataset: Dataset):
        targets = np.array(train_dataset.targets)
        train_indices, val_indices = train_test_split(
            np.arange(len(targets)),
            test_size=self.config.validation_split,
            stratify=targets,
            random_state=self.config.seed
        )
        train_transforms = get_data_transforms(self.config, is_training=True)
        val_transforms = get_data_transforms(self.config, is_training=False)
        train_subset = TransformedSubset(train_dataset, train_indices, train_transforms)
        val_subset = TransformedSubset(train_dataset, val_indices, val_transforms)
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        print(f"Training set size: {len(train_subset)}")
        print(f"Validation set size: {len(val_subset)}")
        return train_loader, val_loader
    def _setup_test_loader(self, test_dataset: Dataset) -> DataLoader:
        test_transforms = get_data_transforms(self.config, is_training=False)
        class TransformedDataset(Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        transformed_test = TransformedDataset(test_dataset, test_transforms)
        return DataLoader(
            transformed_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    def _setup_optimizer(self):
        if self.config.optimizer_type == "Adam":
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "SGD":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    def _default_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        return {"accuracy": accuracy_score(labels, predictions)}
    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        if self.config.use_wandb:
            log_dict = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(log_dict, step=self.global_step)
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        from tqdm import tqdm
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            inputs, labels = batch
            inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.config.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            if self.config.logging_steps > 0 and self.global_step % self.config.logging_steps == 0:
                self._log_metrics({"train_loss": avg_loss}, "train/")
        return total_loss / num_batches
    def evaluate(self, dataloader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(dataloader, desc=f"Evaluating {prefix}", leave=False):
                inputs, labels = batch
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.config.loss_fn(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        metrics = {"loss": avg_loss}
        metrics.update(self.compute_metrics_fn(np.array(all_predictions), np.array(all_labels)))
        self._log_metrics(metrics, f"{prefix}/")
        return metrics
    def train(self) -> Dict[str, float]:
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Start training timer
        self.training_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader, "val")
            print(f"Train Loss: {train_loss:.4f}, Val Metrics: {val_metrics}")
            if self.config.use_early_stopping:
                current_metric = val_metrics.get(self.config.early_stopping_metric)
                if current_metric is not None:
                    if self._is_best_metric(current_metric):
                        self.best_metric = current_metric
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.epochs_without_improvement = 0
                        print("✓ Best model updated!")
                    else:
                        self.epochs_without_improvement += 1
                        print(f"No improvement for {self.epochs_without_improvement} epochs")
                        if self.epochs_without_improvement >= self.config.early_stopping_patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            break
        if self.config.use_early_stopping and self.best_model_state is not None:
            print("Loading best model weights")
            self.model.load_state_dict(self.best_model_state)
        test_metrics = self.evaluate(self.test_loader, "test")
        print(f"\nFinal Test Metrics: {test_metrics}")
        
        # Calculate total training duration
        training_duration = time.time() - self.training_start_time if self.training_start_time else 0
        
        # Send Telegram notification if configured
        self._send_telegram_notification(test_metrics, training_duration)
        
        return test_metrics
    def _is_best_metric(self, current_metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.config.maximize_metric:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def _send_telegram_notification(self, test_metrics: Dict[str, float], training_duration: float) -> None:
        """
        Send a Telegram notification with training results.
        """
        if not self.telegram_notifier:
            return
            
        try:
            # Prepare configuration summary
            config_summary = {
                "Epochs": self.config.num_epochs,
                "Batch Size": self.config.batch_size,
                "Learning Rate": self.config.learning_rate,
                "Optimizer": self.config.optimizer_type,
                "Device": self.config.device,
                "Early Stopping": self.config.use_early_stopping
            }
            
            # Send notification
            success = self.telegram_notifier.send_training_completion_sync(
                model_name=str(self.model),
                training_duration=training_duration,
                final_metrics=test_metrics,
                config_summary=config_summary,
                additional_info=self.config.telegram_additional_info
            )
            
            if not success:
                print("⚠️  Unable to send Telegram notification")
                
        except Exception as e:
            print(f"⚠️  Error sending Telegram notification: {e}")
    
    def save_model(self, suffix: str = "final") -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{str(self.model)}_{suffix}.pth"
        filepath = os.path.join(self.config.output_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
        return filepath
