import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback
import os
import json

class LossPlotter(Callback):

    def __init__(self, save_every_n_epochs=1, save_path="loss_history.json"):
        self.save_every_n_epochs = save_every_n_epochs
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []
        self.train_epochs = []
        self.val_epochs = []

        self._load_history()

    def _load_history(self):
        """Load previously saved loss history if it exists."""
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                data = json.load(f)
                self.train_losses = data.get("train_losses", [])
                self.val_losses = data.get("val_losses", [])
                self.train_epochs = data.get("train_epochs", [])
                self.val_epochs = data.get("val_epochs", [])
            print(f"Loaded previous loss history from {self.save_path}")

    def _save_history(self):
        """Save current loss history."""
        data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_epochs": self.train_epochs,
            "val_epochs": self.val_epochs
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f)
        print(f"Loss history saved to {self.save_path}")

    def on_train_epoch_end(self, trainer, pl_module):
        # With validation enabled, PyTorch Lightning should automatically 
        # aggregate step losses into epoch losses
        train_loss = None
        
        # Try to get epoch-level training loss
        possible_keys = [
            "train_loss_epoch",
            "train/loss_epoch", 
            "train_loss",
            "train/loss"
        ]
        
        # Check callback_metrics first (most reliable)
        for key in possible_keys:
            if key in trainer.callback_metrics:
                train_loss = trainer.callback_metrics[key]
                break
        
        # Fallback to logged_metrics
        if train_loss is None:
            for key in possible_keys:
                if key in trainer.logged_metrics:
                    train_loss = trainer.logged_metrics[key]
                    break
        
        if train_loss is not None:
            if isinstance(train_loss, torch.Tensor):
                loss_value = train_loss.item()
            else:
                loss_value = float(train_loss)
            
            self.train_losses.append(loss_value)
            self.train_epochs.append(trainer.current_epoch)
            print(f"Epoch {trainer.current_epoch}: train_loss={loss_value:.6f}")
            
            if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
                self.save_current_plot(f" (Epoch {trainer.current_epoch})")
                self._save_history()
        else:
            print(f"Epoch {trainer.current_epoch}: Could not find epoch-level train loss")
            print(f"Available callback_metrics: {list(trainer.callback_metrics.keys())}")
            print(f"Available logged_metrics: {list(trainer.logged_metrics.keys())}")

        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            self.save_current_plot(f" (Epoch {trainer.current_epoch})")
            self._save_history()

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val/loss_epoch") or trainer.callback_metrics.get("val/loss")
        if loss is not None:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            self.val_losses.append(loss_val)
            self.val_epochs.append(trainer.current_epoch)
            print(f"Epoch {trainer.current_epoch}: Val loss={loss_val:.6f}")

    def save_current_plot(self, title_suffix=""):
        plt.figure(figsize=(10, 6))
        if self.train_losses:
            plt.plot(self.train_epochs, self.train_losses, label="Train Loss", marker='o')
        if self.val_losses:
            plt.plot(self.val_epochs, self.val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss Over Epochs{title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"loss_curves/loss_plot{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        # ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()
