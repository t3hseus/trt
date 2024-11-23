import logging
import os
import torch
from torch import nn

logging.basicConfig()
logger = logging.getLogger("train")


class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5, suffix: str = ""):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.suffix = suffix
        self.checkpoints = []  # Store (loss, checkpoint_path) tuples
        os.makedirs(save_dir, exist_ok=True)

    def manage_checkpoint(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer | None = None,
            epoch: int = 0,
            loss: float = 1e6
    ):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}{self.suffix}.pth")
        opt_state_dict = None if optimizer is None else optimizer.state_dict()
        # Save the checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt_state_dict,
            'loss': loss,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.checkpoints.append((loss, checkpoint_path))

        self._manage_checkpoints()

    def _manage_checkpoints(self):
        self.checkpoints.sort(key=lambda x: x[0])

        while len(self.checkpoints) > self.max_checkpoints:
            _, worst_checkpoint_path = self.checkpoints.pop()  # Get the worst checkpoint
            os.remove(worst_checkpoint_path)
