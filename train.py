import logging
import os
from os.path import join as pjoin
from typing import Dict

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, Precision, Recall
from tqdm import tqdm

from src.checkpoint_saver import CheckpointManager
from src.dataset import DatasetMode, SPDEventsDataset, collate_fn_with_segmentation_loss
from src.normalization import HitsNormalizer, TrackParamsNormalizer

logging.basicConfig()
logger = logging.getLogger("train")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.random_seed)
    writer = SummaryWriter(log_dir=cfg.hydra_dir)
    out_dir = cfg.hydra_dir
    logger.info("Starting basic objects instantiate")

    hits_norm = HitsNormalizer()  # None
    params_norm = TrackParamsNormalizer()  # None
    train_loader, val_loader = prepare_data(
        hits_norm=hits_norm,
        params_norm=params_norm,
        max_event_tracks=cfg.dataset.max_event_tracks,
        num_events_train=cfg.dataset.train_samples,
        num_events_valid=cfg.dataset.val_samples,
        batch_size=cfg.batch_size,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Device is %s", device)
    model = instantiate(cfg.model).to(device)

    if cfg.resume_from_checkpoint:
        map_location = "cpu" if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

    if cfg.freeze_model:
        model = freeze_model(model, model.params_head)
    criterion = instantiate(cfg.criterion).to(device)
    optimizer = instantiate(cfg.optimizer, model.parameters(), lr=0.0001, weight_decay=0.0001)
    hits_metrics = {
        "accuracy": Accuracy(task="binary", threshold=0.5).to(device),
        "precision": Precision(task="binary", threshold=0.5).to(device),
        "recall": Recall(task="binary", threshold=0.5).to(device),
    }
    train_checkpointer = CheckpointManager(save_dir=cfg.hydra_dir, suffix="_train")
    val_checkpointer = CheckpointManager(save_dir=cfg.hydra_dir, suffix="_val")
    progress_bar = tqdm(range(cfg.num_epochs))
    min_loss_train = min_loss_val = 1e5
    logger.info("Start training... \n")
    for epoch in progress_bar:
        train_loss, min_loss_train = train_epoch(
            train_loader=train_loader,
            model=model,
            num_candidates=model.num_candidates,
            criterion=criterion,
            min_loss_train=min_loss_train,
            optimizer=optimizer,
            writer=writer,
            hits_metrics=hits_metrics,
            device=device,
            epoch=epoch,
            out_dir=out_dir,
            checkpointer=train_checkpointer
        )

        logger.info("Minimal loss is %s", min_loss_train)
        with torch.no_grad():
            val_loss, min_loss_val = val_epoch(
                val_loader=val_loader,
                model=model,
                num_candidates=model.num_candidates,
                criterion=criterion,
                min_loss_val=min_loss_val,
                writer=writer,
                hits_metrics=hits_metrics,
                device=device,
                epoch=epoch,
                out_dir=out_dir,
                checkpointer=val_checkpointer
            )

        progress_bar.set_postfix(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
            }
        )


def prepare_data(
    hits_norm,
    params_norm,
    max_event_tracks: int = 5,
    truncation_length: int = 512,
    num_events_train: int = 1,
    num_events_valid: int = 1,
    batch_size: int = 1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_data = SPDEventsDataset(
        n_samples=num_events_train,
        max_event_tracks=max_event_tracks,
        generate_fixed_tracks_num=False,
        truncation_length=truncation_length,
        hits_normalizer=hits_norm,
        track_params_normalizer=params_norm,
        shuffle=True,
        mode=DatasetMode.train,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_segmentation_loss,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )
    val_data = SPDEventsDataset(
        n_samples=num_events_valid,
        max_event_tracks=max_event_tracks,
        generate_fixed_tracks_num=False,
        truncation_length=truncation_length,
        hits_normalizer=hits_norm,
        track_params_normalizer=params_norm,
        shuffle=False,
        mode=DatasetMode.val,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_segmentation_loss,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_loader, val_loader


def freeze_model(model, head_leave):
    for child in model.children():
        if child == head_leave:
            continue
        for param in child.parameters():
            param.requires_grad = False
    return model


def calc_hits_metrics(
    outputs: Tensor,
    targets: Tensor,
    hits_metrics: dict[str, Metric],
) -> Dict[str, Tensor]:
    res_dict = {}
    outputs = outputs.squeeze(-1)
    for metric in hits_metrics:
        res_dict[metric] = hits_metrics[metric](outputs, targets).to(outputs.device)

    return res_dict


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    writer: SummaryWriter,
    checkpointer: CheckpointManager,
    hits_metrics: dict[str, Metric],
    num_candidates: int = 5,
    epoch: int = 0,
    device: torch.device | str = torch.cuda,
    min_loss_train: float = 1000000.0,
    out_dir: str = "",
) -> tuple[float, float]:
    train_loss = 0.0
    num_train_batches = 0
    model.train()
    for batch in train_loader:
        num_train_batches += 1
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["inputs"].to(device), batch["mask"].to(device))
        loss, loss_components = criterion(
            preds=outputs,
            targets={
                "targets": batch["targets"].to(device),
                "labels": batch["labels"].to(device),
                "end_hits": batch["end_hits"].to(device),
                "hit_labels": batch["hit_labels"].to(device),
            },
            preds_lengths=torch.LongTensor(
                [num_candidates] * batch["inputs"].shape[0]
            ).to(device),
            targets_lengths=batch["n_tracks_per_sample"].to(device),
        )

        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()

        writer.add_scalar(
            "train_loss_batch", loss, epoch * len(train_loader) + num_train_batches
        )

        batch_metrics = calc_hits_metrics(
            outputs=outputs["hit_logits"],
            targets=(batch["hit_labels"].to(device) > -1).to(torch.float),
            hits_metrics=hits_metrics,
        )
        batch_metrics.update(loss_components)
        for metric in batch_metrics:
            writer.add_scalar(
                "train_" + metric,
                batch_metrics[metric],
                epoch * len(train_loader) + num_train_batches,
            )
    checkpointer.manage_checkpoint(model, optimizer, epoch, loss)

    if train_loss / len(train_loader) < min_loss_train:
        min_loss_train = train_loss / len(train_loader)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), pjoin(out_dir, f"trt_hybrid_train_{epoch}.pt"))

    writer.add_scalar("train_loss_epoch", train_loss / len(train_loader), epoch)
    for metric in hits_metrics:
        writer.add_scalar(
            f"train_{metric}_epoch",
            hits_metrics[metric].compute(),
            epoch,
        )

    return train_loss, min_loss_train


def val_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    writer: SummaryWriter,
    checkpointer: CheckpointManager,
    hits_metrics: dict[str, Metric],
    num_candidates: int = 5,
    epoch: int = 0,
    device: torch.device | str = torch.cuda,
    min_loss_val: float = 1000000.0,
    out_dir: str = "",
) -> tuple[float, float]:
    val_loss = 0.0
    num_val_batches = 0
    model.eval()
    for batch in val_loader:
        num_val_batches += 1
        outputs = model(batch["inputs"].to(device), batch["mask"].to(device))
        loss, loss_components = criterion(
            preds=outputs,
            targets={
                "targets": batch["targets"].to(device),
                "labels": batch["labels"].to(device),
                "end_hits": batch["end_hits"].to(device),
                "hit_labels": batch["hit_labels"].to(device),
            },
            preds_lengths=torch.LongTensor(
                [num_candidates] * batch["inputs"].shape[0]
            ).to(device),
            targets_lengths=batch["n_tracks_per_sample"].to(device),
        )
        val_loss += loss.detach().item()

        writer.add_scalar(
            "val_loss_batch", loss, epoch * len(val_loader) + num_val_batches
        )

        batch_metrics = calc_hits_metrics(
            outputs=outputs["hit_logits"],
            targets=(batch["hit_labels"].to(device) > -1).to(torch.float),
            hits_metrics=hits_metrics,
        )
        batch_metrics.update(loss_components)
        for metric in batch_metrics:
            writer.add_scalar(
                "val_" + metric,
                batch_metrics[metric],
                epoch * len(val_loader) + num_val_batches,
            )
    checkpointer.manage_checkpoint(model, optimizer=None, epoch=epoch, loss=loss)
    if val_loss / len(val_loader) < min_loss_val:
        min_loss_val = val_loss / len(val_loader)

    writer.add_scalar("val_loss_epoch", val_loss / len(val_loader), epoch)
    for metric in hits_metrics:
        writer.add_scalar(
            f"val_{metric}_epoch",
            hits_metrics[metric].compute(),
            epoch,
        )

    return val_loss, min_loss_val


if __name__ == "__main__":
    main()
