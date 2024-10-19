from enum import IntEnum
from typing import List, Optional, TypedDict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_generation import ArrayN, ArrayNx3, SPDEventGenerator
from .normalization import (ConstraintsNormalizer, NormTParamsArr, TParamsArr,
                            TrackParamsNormalizer)


class DatasetMode(IntEnum):
    train = 0
    val = 1
    test = 2


class DatasetSample(TypedDict):
    hits: ArrayNx3[np.float32]
    hit_labels: ArrayN[np.int32]
    params: Union[TParamsArr, NormTParamsArr]
    param_labels: ArrayN[np.int32]
    mask: ArrayN[np.float32]
    orig_params: Union[TParamsArr, NormTParamsArr]


class BatchSample(TypedDict):
    inputs: torch.FloatTensor
    mask: torch.FloatTensor
    targets: torch.FloatTensor
    orig_params: torch.FloatTensor
    n_tracks_per_sample: torch.LongTensor


class BatchSampleWithLogits(BatchSample):
    labels: torch.LongTensor


class BatchSampleWithHitLabels(BatchSampleWithLogits):
    hit_labels: torch.LongTensor

class SPDEventsDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        max_event_tracks: int = 10,
        generate_fixed_tracks_num: bool = True,
        add_fakes: bool = True,
        shuffle: bool = True,
        detector_eff: float = 0.98,
        hits_normalizer: Optional[ConstraintsNormalizer] = None,
        track_params_normalizer: Optional[TrackParamsNormalizer] = None,
        truncation_length: Optional[int] = None,
        fakes_label: int = -1,
        # padding_label: int = -1,
        mode: DatasetMode = DatasetMode.train,
    ):
        self._n_samples = n_samples
        self._max_event_tracks = max_event_tracks
        self._add_fakes = add_fakes
        self._fakes_label = fakes_label
        # self._padding_label = padding_label
        self._shuffle = shuffle
        self._generate_fixed_tracks_num = generate_fixed_tracks_num

        if truncation_length is not None and truncation_length > 0:
            self.truncation_length = truncation_length
        else:
            self.truncation_length = None

        self.spd_gen = SPDEventGenerator(
            generate_fixed_tracks_num=generate_fixed_tracks_num,
            max_event_tracks=max_event_tracks,
            add_fakes=add_fakes,
            detector_eff=detector_eff,
        )
        self.hits_normalizer = hits_normalizer
        self.track_params_normalizer = track_params_normalizer
        # get initial random seed for reproducibility
        # mode helps to ensure that datasets don't intersect
        self._initial_seed = np.random.get_state()[1][mode]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> DatasetSample:
        # prevent dataset from generation of new samples each epoch
        np.random.seed(self._initial_seed + idx)
        # generate sample
        event = self.spd_gen.generate_spd_event()

        # get all hits including fakes
        if self._add_fakes:
            hits = np.vstack([event.hits, event.fakes])
            hit_labels = np.hstack(
                # -1 - label for fakes
                [event.track_ids, np.full(len(event.fakes), self._fakes_label)]
            )
        else:
            hits = event.hits
            hit_labels = event.track_ids

        # collect tracks parameters
        # vx, vy, vz, phi, theta, pt, charge
        # all params are normalized
        # charge -> categorical charge
        # charge=-1 -> (1  0) charge=1 -> (0  1);
        # theta -> theta raw, i.e. before arccos

        # params_shape = (self._max_event_tracks, 7)
        params_shape = (len(event.track_params), 7)

        params = np.zeros(params_shape, dtype=np.float32)
        orig_params = np.zeros(params_shape, dtype=np.float32)
        # TODO: use only generated number of tracks without padding
        # param_labels = np.full(
        #     self._max_event_tracks, self._padding_label, dtype=np.int32
        # )
        param_labels = np.zeros(len(event.track_params.items()), dtype=np.int32)

        for i, (track_id, track_params) in enumerate(event.track_params.items()):
            # normalize track parameters if needed
            if self.track_params_normalizer:
                params[i] = self.track_params_normalizer.normalize(
                    vx=event.vertex.x,
                    vy=event.vertex.y,
                    vz=event.vertex.z,
                    pt=track_params.pt,
                    phi=track_params.phi,
                    theta=track_params.theta,
                    charge=track_params.charge,
                )
                orig_params[i] = self.track_params_normalizer.denormalize(
                    params[i], is_charge_categorical=True
                )
            else:
                params[i][:3] = event.vertex.numpy
                params[i][3:] = track_params.numpy
                orig_params[i] = params[i]
            param_labels[i] = track_id

        # shuffle data before output
        if self._shuffle:
            shuffle_idx = np.random.permutation(len(hits))
            hits = hits[shuffle_idx]
            hit_labels = hit_labels[shuffle_idx]
            # shuffle params without shuffling padding
            shuffle_idx = np.random.permutation(event.n_tracks)
            params[: event.n_tracks] = params[shuffle_idx]
            orig_params[: event.n_tracks] = orig_params[shuffle_idx]
            param_labels[: event.n_tracks] = param_labels[shuffle_idx]

        # data normalization
        if self.hits_normalizer:
            hits = self.hits_normalizer(hits)

        if self.truncation_length is not None:
            # truncate inputs
            hits = hits[: self.truncation_length]
            hit_labels = hit_labels[: self.truncation_length]

        return DatasetSample(
            hits=hits,
            hit_labels=hit_labels,
            params=params,
            orig_params=orig_params,
            param_labels=param_labels,
            mask=np.ones(len(hits), dtype=bool),
        )


def collate_fn_with_class_loss(samples: List[DatasetSample]) -> BatchSample:
    max_n_hits = max([len(sample["hits"]) for sample in samples])
    n_tracks_per_sample = [len(sample["params"]) for sample in samples]
    max_n_tracks = max(n_tracks_per_sample)
    batch_size = len(samples)
    n_features = samples[0]["hits"].shape[-1]

    batch_inputs = np.zeros((batch_size, max_n_hits, n_features), dtype=np.float32)
    batch_mask = np.zeros((batch_size, max_n_hits), dtype=bool)
    # params have the fixed size - MAX_TRACKS x N_PARAMS
    target_shape = (batch_size, max_n_tracks, samples[0]["params"].shape[1])
    batch_params = np.zeros(target_shape, dtype=np.float32)
    batch_orig_params = np.zeros(target_shape, dtype=np.float32)
    batch_labels = np.ones((batch_size, max_n_tracks), dtype=np.int32)

    for i, sample in enumerate(samples):
        batch_inputs[i, : len(sample["hits"])] = sample["hits"]
        batch_mask[i, : len(sample["hits"])] = sample["mask"]
        batch_params[i, : len(sample["params"])] = sample["params"]
        batch_labels[i, : len(sample["params"])] = 0  # class 0 is gt, 1 is no-object
        batch_orig_params[i, : len(sample["orig_params"])] = sample["orig_params"]

    return BatchSampleWithLogits(
        inputs=torch.tensor(batch_inputs, dtype=torch.float),
        mask=torch.from_numpy(batch_mask),
        targets=torch.from_numpy(batch_params),
        orig_params=torch.from_numpy(batch_orig_params),
        n_tracks_per_sample=torch.LongTensor(n_tracks_per_sample),
        labels=torch.from_numpy(batch_labels).to(torch.long),
    )


def collate_fn_with_segment_loss(samples: List[DatasetSample]) -> BatchSample:
    max_n_hits = max([len(sample["hits"]) for sample in samples])
    n_tracks_per_sample = [len(sample["params"]) for sample in samples]
    max_n_tracks = max(n_tracks_per_sample)
    batch_size = len(samples)
    n_features = samples[0]["hits"].shape[-1]

    batch_inputs = np.zeros(
        (batch_size, max_n_hits, n_features), dtype=np.float32
    )
    batch_mask = np.zeros((batch_size, max_n_hits), dtype=bool)
    # params have the fixed size - MAX_TRACKS x N_PARAMS
    target_shape = (batch_size, max_n_tracks, samples[0]["params"].shape[1])
    batch_params = np.zeros(target_shape, dtype=np.float32)
    batch_orig_params = np.zeros(target_shape, dtype=np.float32)
    batch_labels = np.ones((batch_size, max_n_tracks), dtype=np.int32)
    batch_hit_labels = np.ones((batch_size, max_n_hits), dtype=np.int32)

    for i, sample in enumerate(samples):
        batch_inputs[i, :len(sample["hits"])] = sample["hits"]
        batch_hit_labels[i, :len(sample["hits"])] = sample["hit_labels"] > -1
        batch_mask[i, :len(sample["hits"])] = sample["mask"]
        batch_params[i, :len(sample["params"])] = sample["params"]
        batch_labels[i, :len(sample["params"])] = 0  # class 0 is gt, 1 is no-object
        batch_orig_params[
        i, :len(sample["orig_params"])
        ] = sample["orig_params"]

    return BatchSampleWithHitLabels(
        inputs=torch.tensor(batch_inputs, dtype=torch.float),
        mask=torch.from_numpy(batch_mask),
        targets=torch.from_numpy(batch_params),
        orig_params=torch.from_numpy(batch_orig_params),
        n_tracks_per_sample=torch.LongTensor(n_tracks_per_sample),
        labels=torch.from_numpy(batch_labels).to(torch.long),
        hit_labels=torch.from_numpy(batch_hit_labels).to(torch.long),
    )
