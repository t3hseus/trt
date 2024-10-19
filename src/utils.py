import re
from pathlib import Path
from typing import Union

CHECKPOINT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")


def get_latest_checkpoint(checkpoint_dir: Union[str, Path], model_name: str):
    checkpoint_dir = Path(checkpoint_dir)
    latest_ckpt = None
    max_epoch = -1
    max_step = -1

    # Iterate through all files in the checkpoint directory
    for ckpt_file in checkpoint_dir.glob(f"{model_name}/version_*/*.ckpt"):
        match = CHECKPOINT_PATTERN.search(ckpt_file.name)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))

            # Update the latest checkpoint if this one has a higher epoch (or higher step if same epoch)
            if (epoch > max_epoch) or (epoch == max_epoch and step > max_step):
                max_epoch = epoch
                max_step = step
                latest_ckpt = ckpt_file

    return latest_ckpt
