from src.training import TrainModel
from src.logging_utils import setup_logger
from src.dataset import SPDEventsDataset, DatasetMode, collate_fn
from src.normalization import ConstraintsNormalizer, TrackParamsNormalizer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Optional, Callable, Union
from torch import optim
from torch import nn
from argparse import ArgumentParser
import pytorch_lightning as pl
import torchmetrics as tm
import gin
import logging
import os
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


argparser = ArgumentParser()
argparser.add_argument(
    "--config", type=str,
    help="Path to the config file to use."
)
argparser.add_argument(
    "--log", type=str, default="INFO",
    choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
    help='Level of logging'
)


logging.basicConfig()
LOGGER = logging.getLogger("train")


@gin.configurable
def experiment(
        model: nn.Module,
        criterion: nn.Module,
        metrics: List[Union[Callable, tm.Metric]],
        optimizer: optim.Optimizer,
        train_samples: int = 1000,
        val_samples: int = 10,
        max_event_tracks: int = 10,
        detector_efficiency: float = 0.98,
        truncation_length: int = 512,
        hits_normalizer: ConstraintsNormalizer = ConstraintsNormalizer(),
        track_params_normalizer: TrackParamsNormalizer = TrackParamsNormalizer(),
        num_epochs: int = 10,
        train_batch_size: int = 8,
        val_batch_size: int = 16,
        random_seed: int = 42,
        logging_dir: str = "experiment_logs",
        resume_from_checkpoint: Optional[str] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
):
    os.makedirs(logging_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(logging_dir, name=model.__class__.__name__)
    setup_logger(LOGGER, tb_logger.log_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tb_logger.log_dir}",
        filename=f'{{epoch}}-{{step}}'
    )

    with open(os.path.join(tb_logger.log_dir, "train_config.cfg"), "w") as f:
        f.write(gin.config_str())

    LOGGER.info(f"Log directory {tb_logger.log_dir}")
    LOGGER.info(
        "GOT config: \n======config======\n "
        f"{gin.config_str()} "
        "\n========config======="
    )

    LOGGER.info(f"Setting random seed to {random_seed}")
    pl.seed_everything(random_seed)

    LOGGER.info("Preparing datasets for training and validation")
    train_data = SPDEventsDataset(
        n_samples=train_samples,
        max_event_tracks=max_event_tracks,
        truncation_length=truncation_length,
        detector_eff=detector_efficiency,
        hits_normalizer=hits_normalizer,
        track_params_normalizer=track_params_normalizer,
        mode=DatasetMode.train
    )
    val_data = SPDEventsDataset(
        n_samples=val_samples,
        max_event_tracks=max_event_tracks,
        truncation_length=truncation_length,
        detector_eff=detector_efficiency,
        hits_normalizer=hits_normalizer,
        track_params_normalizer=track_params_normalizer,
        mode=DatasetMode.val
    )

    # check both determinism and test != train
    assert train_data[0]["params"].mean() == train_data[0]["params"].mean()
    assert val_data[0]["params"].mean() == val_data[0]["params"].mean()
    assert train_data[0]["params"].mean() != val_data[0]["params"].mean()

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    LOGGER.info('Creating model for training')
    # trt_model = TrainModel(
    #     model=model,
    #     criterion=criterion,
    #     metrics=metrics,
    #     optimizer=optimizer
    # )
    # LOGGER.info(trt_model)

    if resume_from_checkpoint is not None:
        LOGGER.info(f"Resuming from checkpoint {resume_from_checkpoint}")

    # trainer = pl.Trainer(
    #     max_epochs=num_epochs,
    #     deterministic=True,
    #     accelerator="auto",
    #     logger=tb_logger,
    #     callbacks=[checkpoint_callback]
    # )
    # trainer.fit(
    #     model=trt_model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    # )


def main():
    args = argparser.parse_args()
    LOGGER.setLevel(getattr(logging, args.log))
    gin.parse_config(open(args.config))
    experiment()
    LOGGER.info("End of training")


if __name__ == "__main__":
    main()
