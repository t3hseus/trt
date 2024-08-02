from typing import Callable, Dict, List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torch import optim

from .dataset import BatchSample


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
        orig_params = batch.pop("orig_params")
        preds = self.forward(batch)
        loss = self.criterion(preds, targets)
        metric_vals = self._calc_metrics(preds, targets)
        return {"loss": loss, **metric_vals}

    def training_step(self, batch: BatchSample, batch_idx: int):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f"train_{k}": v for k, v in result_dict.items()}
        self.log_dict(tqdm_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return tqdm_dict["train_loss"]

    def validation_step(self, batch: BatchSample, batch_idx: int):
        result_dict = self._forward_batch(batch)
        tqdm_dict = {f"val_{k}": v for k, v in result_dict.items()}
        self.log_dict(tqdm_dict, prog_bar=True, sync_dist=True)
        return tqdm_dict["val_loss"]

    def configure_optimizers(self):
        return self.optimizer  # (self.parameters())
