from typing import List

import numpy as np
import torch

from src.dataset import BatchSample, DatasetSample


def collate_fn(samples: List[DatasetSample]) -> BatchSample:
    """
    #TODO: add special collate without padded track params in targets

    idx = matched(preds, targets) -> list[tuple(int, int)]
    idx = {(1, 0), (3, 2), (4, 1)} # B-b, D-d, E-e

    clf_targets = {0, 1, 0, 1, 1}
    pred_targets = {1, 1, 0, 0, 1}

    clf_loss = cross_entropy(clf_targets, pred_targets)
    params_loss = mae(pred[idx], targets)
    hits_loss = mae(generate_hits(pred[idx]), generate_hits(targets))

    loss = alpha * clf_loss + beta * params_loss + gamma * hits_loss
    """
    maxlen = max([len(sample["hits"]) for sample in samples])
    batch_size = len(samples)
    n_features = samples[0]["hits"].shape[-1]

    batch_inputs = np.zeros((batch_size, maxlen, n_features), dtype=np.float32)
    batch_mask = np.zeros((batch_size, maxlen), dtype=bool)
    # params have the fixed size - MAX_TRACKS x N_PARAMS
    batch_targets = np.zeros(
        (batch_size, *samples[0]["params"].shape), dtype=np.float32
    )
    batch_orig_params = np.zeros(
        (batch_size, *samples[0]["orig_params"].shape), dtype=np.float32
    )

    for i, sample in enumerate(samples):
        batch_inputs[i, : len(sample["hits"])] = sample["hits"]
        batch_mask[i, : len(sample["hits"])] = sample["mask"]
        batch_targets[i] = sample["params"]
        batch_orig_params[i] = sample["orig_params"]

    return BatchSample(
        inputs=torch.from_numpy(batch_inputs),
        mask=torch.from_numpy(batch_mask),
        targets=torch.from_numpy(batch_targets),
        orig_params=torch.from_numpy(batch_orig_params),
    )


def collate_fn_for_set_loss(samples: List[DatasetSample]) -> BatchSample:
    max_n_hits = max([len(sample["hits"]) for sample in samples])
    n_tracks_per_sample = [len(sample["params"]) for sample in samples]
    max_n_tracks = max(n_tracks_per_sample)
    batch_size = len(samples)
    n_features = samples[0]["hits"].shape[-1]

    batch_inputs = np.zeros((batch_size, max_n_hits, n_features), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_n_hits), dtype=bool)
    # params have the fixed size - MAX_TRACKS x N_PARAMS
    target_shape = (batch_size, max_n_tracks, samples[0]["params"].shape[1])
    batch_targets = np.zeros(target_shape, dtype=np.float32)
    batch_orig_params = np.zeros(target_shape, dtype=np.float32)

    for i, sample in enumerate(samples):
        batch_inputs[i, : len(sample["hits"])] = sample["hits"]
        batch_mask[i, : len(sample["hits"])] = sample["mask"]
        batch_targets[i, : len(sample["params"])] = sample["params"]
        batch_orig_params[i, : len(sample["orig_params"])] = sample["orig_params"]

    return BatchSample(
        inputs=torch.from_numpy(batch_inputs),
        mask=torch.from_numpy(batch_mask),
        targets=torch.from_numpy(batch_targets),
        orig_params=torch.from_numpy(batch_orig_params),
        n_tracks_per_sample=torch.LongTensor(n_tracks_per_sample),
    )
