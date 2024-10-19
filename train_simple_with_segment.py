import os
from datetime import datetime
from os.path import join as pjoin

import torch
import torchmetrics
from pytorch_lightning import seed_everything
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric, Precision
from tqdm import tqdm

from src.dataset import (DatasetMode, SPDEventsDataset,
                         collate_fn_with_segment_loss)
from src.loss import TRTHungarianLoss, TRTLossWithSegment
from src.metrics import vertex_distance
from src.models.model_with_segmentation import TRTWithSegmentation
from src.normalization import ConstraintsNormalizer, TrackParamsNormalizer

seed_everything(13)

MAX_EVENT_TRACKS = 5
NUM_CANDIDATES = MAX_EVENT_TRACKS * 10
TRUNCATION_LENGTH = 1024
BATCH_SIZE = 32
NUM_EVENTS_TRAIN = 50000
NUM_EVENTS_VALID = 1000
INTERMEDIATE = False


def main():
    writer = SummaryWriter()
    hits_norm = ConstraintsNormalizer()
    # hits_norm = None
    params_norm = TrackParamsNormalizer()
    # params_norm = None
    out_dir = pjoin(
        r"E:\projects\trt\weights",
        datetime.today().strftime("%Y-%m-%d"),
    )

    train_loader, val_loader = prepare_data(
        hits_norm=hits_norm,
        params_norm=params_norm,
        max_event_tracks=MAX_EVENT_TRACKS,
        num_events_train=NUM_EVENTS_TRAIN,
        num_events_valid=NUM_EVENTS_VALID,
        batch_size=BATCH_SIZE,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device is", device)
    model = TRTWithSegmentation(
        num_candidates=NUM_CANDIDATES,
        n_points=TRUNCATION_LENGTH,
        num_out_params=7,
        return_intermediate=INTERMEDIATE,
    ).to(device)
    # model.load_state_dict(torch.load(r"weights\best\trt_hybrid_val.pt", weights_only=True))

    criterion = TRTLossWithSegment(
        weights=(0.33, 0.33, 0.33, 0.03), intermediate=INTERMEDIATE
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.00001)

    progress_bar = tqdm(range(5000))
    min_loss_train = min_loss_val = 1e5
    for epoch in progress_bar:
        train_loss, min_loss_train = train_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            min_loss_train=min_loss_train,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
            out_dir=out_dir,
        )
        val_loss, min_loss_val = val_epoch(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            min_loss_val=min_loss_val,
            writer=writer,
            device=device,
            epoch=epoch,
            out_dir=out_dir,
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
    max_event_tracks: int = MAX_EVENT_TRACKS,
    truncation_length: int = TRUNCATION_LENGTH,
    num_events_train: int = NUM_EVENTS_TRAIN,
    num_events_valid: int = NUM_EVENTS_VALID,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_data = SPDEventsDataset(
        max_event_tracks=max_event_tracks,
        generate_fixed_tracks_num=False,
        hits_normalizer=hits_norm,
        track_params_normalizer=params_norm,
        shuffle=True,
        truncation_length=truncation_length,
        n_samples=num_events_train,
        mode=DatasetMode.train,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_segment_loss,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    val_data = SPDEventsDataset(
        n_samples=num_events_valid,
        max_event_tracks=max_event_tracks,
        truncation_length=truncation_length,
        generate_fixed_tracks_num=False,
        hits_normalizer=hits_norm,
        track_params_normalizer=params_norm,
        mode=DatasetMode.val,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_segment_loss,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_loader, val_loader


def calc_metrics(outputs, batch, device, hit_metrics: dict[str, Metric]):
    params_only_dist = TRTHungarianLoss(weights=(1, 0.0, 0.0), intermediate=False).to(
        device
    )
    res_dict = {}
    params = outputs["params"][-1].to(device) if INTERMEDIATE else outputs["params"]
    logits = outputs["logits"][-1].to(device) if INTERMEDIATE else outputs["logits"]
    vertex = outputs["vertex"]
    hits_classes = outputs["hit_logits"][:, :, 1].to(device)
    for metric in hit_metrics:
        res_dict[metric] = hit_metrics[metric](
            hits_classes, batch["hit_labels"].to(device)
        )
    vertex_dist = vertex_distance(vertex, batch["targets"].to(device))
    last_out_params_dist = params_only_dist(
        preds={"params": params, "logits": logits, "vertex": vertex},
        targets={
            "targets": batch["targets"].to(device),
            "labels": batch["labels"].to(device),
        },
        preds_lengths=torch.LongTensor(
            [NUM_CANDIDATES]
            * params.shape[-3]
            # if we have intermediate losses, shape is 4 dim, else 3 dim
        ).to(device),
        targets_lengths=batch["n_tracks_per_sample"].to(device),
    )
    return {"vertex_dist": vertex_dist, "params_dist": last_out_params_dist, **res_dict}


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    writer,
    epoch: int = 0,
    device: torch.device | str = torch.cuda,
    min_loss_train: float = 1000000.0,
    out_dir: str = "",
) -> tuple[float, float]:
    train_loss = 0.0
    num_train_batches = 0
    hits_metrics = {
        "accuracy": torchmetrics.classification.Accuracy(
            task="binary", threshold=0.5
        ).to(device),
        "preccision": torchmetrics.classification.Precision(
            task="binary", threshold=0.5
        ).to(device),
        "recall": torchmetrics.classification.Recall(task="binary", threshold=0.5).to(
            device
        ),
    }

    for batch in train_loader:
        num_train_batches += 1
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["inputs"].to(device), batch["mask"].to(device))
        loss = criterion(
            preds=outputs,
            targets={
                "targets": batch["targets"].to(device),
                "labels": batch["labels"].to(device),
                "hit_labels": batch["hit_labels"].to(device),
            },
            preds_lengths=torch.LongTensor(
                [NUM_CANDIDATES]
                * outputs["params"].shape[-3]
                # if we have intermediate losses, shape is 4 dim, else 3 dim
            ).to(device),
            targets_lengths=batch["n_tracks_per_sample"].to(device),
        )

        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()

        batch_metrics = calc_metrics(outputs, batch, device, hits_metrics)
        writer.add_scalar(
            "train_loss_batch", loss, epoch * len(train_loader) + num_train_batches
        )
        for metric in batch_metrics:
            writer.add_scalar(
                "train_" + metric,
                batch_metrics[metric],
                epoch * len(train_loader) + num_train_batches,
            )

    if train_loss < min_loss_train:
        min_loss_train = train_loss
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), pjoin(out_dir, "trt_hybrid_train.pt"))

    writer.add_scalar("train_loss_epoch", train_loss / len(train_loader), epoch)
    for metric in hits_metrics:
        writer.add_scalar(
            f"train_{metric}_epoch",
            hits_metrics[metric].compute() / len(train_loader),
            epoch,
        )
    return train_loss, min_loss_train


def val_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    writer,
    epoch: int = 0,
    device: torch.device | str = torch.cuda,
    min_loss_val: float = 1000000.0,
    out_dir="",
) -> tuple[float, float]:
    val_loss = 0
    num_val_batches = 0
    model.eval()
    hits_metrics = {
        "accuracy": torchmetrics.classification.Accuracy(
            task="binary", threshold=0.5
        ).to(device),
        "preccision": torchmetrics.classification.Precision(
            task="binary", threshold=0.5
        ).to(device),
        "recall": torchmetrics.classification.Recall(task="binary", threshold=0.5).to(
            device
        ),
    }
    for batch in val_loader:
        num_val_batches += 1
        outputs = model(batch["inputs"].to(device), batch["mask"].to(device))
        loss = criterion(
            preds=outputs,
            targets={
                "targets": batch["targets"].to(device),
                "labels": batch["labels"].to(device),
                "hit_labels": batch["hit_labels"].to(device),
            },
            preds_lengths=torch.LongTensor(
                [MAX_EVENT_TRACKS]
                * outputs["params"].shape[-3]
                # if we have intermediate losses, shape is 4 dim, else 3 dim
            ).to(device),
            targets_lengths=batch["n_tracks_per_sample"].to(device),
        )
        val_loss += loss.detach().item()

        batch_metrics = calc_metrics(outputs, batch, device, hits_metrics)
        writer.add_scalar(
            "val_loss_batch", loss, epoch * len(val_loader) + num_val_batches
        )
        for metric in batch_metrics:
            writer.add_scalar(
                "val_" + metric,
                batch_metrics[metric],
                epoch * len(val_loader) + num_val_batches,
            )

    if val_loss < min_loss_val:
        min_loss_val = val_loss
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), pjoin(out_dir, "trt_hybrid_val.pt"))

    writer.add_scalar("val_loss_epoch", val_loss / len(val_loader), epoch)
    for metric in hits_metrics:
        writer.add_scalar(
            f"train_{metric}_epoch",
            hits_metrics[metric].compute() / len(val_loader),
            epoch,
        )
    return val_loss, min_loss_val


if __name__ == "__main__":
    main()
