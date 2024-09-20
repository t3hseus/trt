import io
import torch
import numpy as np
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from typing import Optional, Callable, Dict, List, Union
from typing_extensions import Self
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path
from PIL import Image
from torch import optim

from .dataset import BatchSample
from .visualization import draw_event
from .utils import get_latest_checkpoint


class TrainModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: List[Union[Callable, tm.Metric]],
        optimizer: optim.Optimizer,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.validation_step_outputs = []

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        strict: Optional[bool] = None,
    ) -> Self:
        """Overrides the original LightningModule method to load from the checkpoint dir.

        Args:
            checkpoint_path: Path to the directory with checkpoint
            strict: Whether to strictly enforce that the keys in :attr:`checkpoint_path` of the
                original `load_from_checkpoint` method match the keys returned by this module's state dict.
                Defaults to ``True`` unless ``LightningModule.strict_loading`` is set, in which case
                it defaults to the value of ``LightningModule.strict_loading``.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise ValueError("checkpoint_dir must be a real directory")

        config_dir = str((checkpoint_dir / ".hydra").absolute())

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config.yaml")

        # initialize main blocks of the model
        model = instantiate(cfg.model)
        criterion = instantiate(cfg.criterion)
        metrics = [instantiate(metric) for metric in cfg.metrics]
        optimizer = instantiate(cfg.optimizer, params=model.parameters())

        # get the latest checkpoint path
        checkpoint_path = get_latest_checkpoint(
            checkpoint_dir=checkpoint_dir, model_name=model.__class__.__name__
        )

        return super().load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            strict=strict,
        )

    def forward(self, inputs: Dict[str, torch.FloatTensor]):
        """
        # Arguments
            inputs (dict): kwargs dict with model inputs
        """
        return self.model(**inputs)

    def _calc_metrics(self, predictions, targets):
        metric_vals = {}
        for metric in self.metrics:
            metric_vals[metric.__name__] = metric(predictions, targets)
        return metric_vals

    def _forward_batch(self, batch: BatchSample):
        targets = batch.pop("targets")
        _ = batch.pop("orig_params")
        preds = self.forward(batch)
        loss = self.criterion(preds, targets)

        metric_vals = self._calc_metrics(preds, targets)
        return {"loss": loss, "prediction": preds, "target": targets, **metric_vals}

    def training_step(self, batch: BatchSample, batch_idx: int):
        result_dict = self._forward_batch(batch)
        _, _ = result_dict.pop("prediction"), result_dict.pop("target")
        tqdm_dict = {f"train_{k}": v for k, v in result_dict.items()}
        self.log_dict(tqdm_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return tqdm_dict["train_loss"]

    def validation_step(self, batch: BatchSample, batch_idx: int):
        result_dict = self._forward_batch(batch)
        # save results of validation for on_validation_epoch_end callback
        self.validation_step_outputs.append(
            {
                "prediction": result_dict.pop("prediction"),
                "target": result_dict.pop("target"),
            }
        )
        tqdm_dict = {f"val_{k}": v for k, v in result_dict.items()}
        self.log_dict(tqdm_dict, prog_bar=True, sync_dist=True)
        return tqdm_dict["val_loss"]

    def configure_optimizers(self):

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.1, total_steps=30000
        )
        return [self.optimizer], [scheduler]  # (self.parameters())

    def save_prediction_figure_to_tensorboard(
        self,
        pred_orig_figure: go.Figure,
        tag: str = "Event Reconstruction Visualization",
    ):
        # Convert the Plotly figure to a PNG image
        png_bytes = pred_orig_figure.to_image(format="png", scale=3)

        # Load the image with PIL
        image = Image.open(io.BytesIO(png_bytes))

        # Convert the image to a NumPy array (required by Matplotlib's imshow)
        image_array = np.array(image)

        # Create a Matplotlib figure and add the image
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_array)
        ax.axis("off")  # Disable axes for better display
        plt.tight_layout()

        # Log the Matplotlib figure to TensorBoard
        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

        # Close the Matplotlib figure to free memory
        plt.close(fig)

    def on_validation_epoch_end_1(self):
        # take first event from first batch for validation
        sample_idx = 0
        event_idx = 7
        sample_for_visualization = self.validation_step_outputs[sample_idx]
        # get target event plot on first epoch
        # no need to re-draw it every time
        print("start_vis")
        if self.current_epoch == 0:
            hits, track_ids = self.criterion.event_generator(
                pred_params=sample_for_visualization["target"][event_idx][:, :-1],
                pred_charges=sample_for_visualization["target"][event_idx][:, -1],
                from_targets=True,
            )
            orig_event_fig = draw_event(hits, track_ids)
            self.save_prediction_figure_to_tensorboard(
                orig_event_fig, tag=f"Batch {sample_idx}, event {event_idx} original"
            )

        # get reconstructed event plot
        hits, track_ids = self.criterion.hits_generator(
            pred_params=sample_for_visualization["prediction"]["params"][event_idx],
            pred_charges=sample_for_visualization["prediction"]["logits"][event_idx],
        )
        pred_event_fig = draw_event(hits, track_ids)
        self.save_prediction_figure_to_tensorboard(
            pred_event_fig, tag=f"Batch {sample_idx}, event {event_idx} prediction"
        )
        print("HERE")
        self.validation_step_outputs.clear()  # free memory
