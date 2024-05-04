import numpy as np
from enum import IntEnum
from typing import Optional, Dict
from torch.utils.data import Dataset
from .data_generation import SPDEventGenerator
from .normalization import ConstraintsNormalizer, TrackParamsNormalizer


class DatasetMode(IntEnum):
    train = 0
    val = 1
    test = 2


class SPDEventsDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        max_event_tracks: int = 10,
        add_fakes: bool = True,
        shuffle: bool = True,
        detector_eff: float = 0.98,
        hits_normalizer: Optional[ConstraintsNormalizer] = None,
        track_params_normalizer: Optional[TrackParamsNormalizer] = None,
        fakes_label: int = -1,
        padding_label: int = -1,
        mode: DatasetMode = DatasetMode.train
    ):
        self._n_samples = n_samples
        self._max_event_tracks = max_event_tracks
        self._add_fakes = add_fakes
        self._fakes_label = fakes_label
        self._padding_label = padding_label
        self._shuffle = shuffle
        self.spd_gen = SPDEventGenerator(
            max_event_tracks=max_event_tracks,
            add_fakes=add_fakes,
            detector_eff=detector_eff)
        self.hits_normalizer = hits_normalizer
        self.track_params_normalizer = track_params_normalizer
        # get initial random seed for reproducibility
        # mode helps to ensure that datasets don't intersect
        self._initial_seed = np.random.get_state()[1][mode]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # prevent dataset from generation of new samples each epoch
        np.random.seed(self._initial_seed + idx)
        # generate sample
        event = self.spd_gen.generate_spd_event()

        # get all hits including fakes
        if self._add_fakes:
            hits = np.vstack([event.hits, event.fakes])
            hit_labels = np.hstack(
                # -1 - label for fakes
                [event.track_ids, np.full(len(event.fakes), self._fakes_label)])
        else:
            hits = event.hits
            hit_labels = event.track_ids

        # collect tracks parameters
        # vx, vy, vz, phi, theta, pt, charge
        # all params are normalized
        # charge -> categorical charge
        # charge=-1 -> (1  0) charge=1 -> (0  1);
        # theta -> theta raw, i.e. before arccos
        if self.track_params_normalizer:
            # charge -> categorical feature
            # charge=`-1` -> `(1,  0)` charge=`1` -> `(0, 1)`;
            params_shape = (self._max_event_tracks, 8)
        else:
            params_shape = (self._max_event_tracks, 7)

        params = np.zeros(params_shape, dtype=np.float32)
        param_labels = np.full(self._max_event_tracks,
                               self._padding_label, dtype=np.int32)
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
                    charge=track_params.charge
                )
            else:
                params[i][:3] = event.vertex.numpy
                params[i][3:] = track_params.numpy
            param_labels[i] = track_id

        # shuffle data before output
        if self._shuffle:
            shuffle_idx = np.random.permutation(len(hits))
            hits = hits[shuffle_idx]
            hit_labels = hit_labels[shuffle_idx]
            # shuffle params without shuffling padding
            shuffle_idx = np.random.permutation(event.n_tracks)
            params[:event.n_tracks] = params[shuffle_idx]
            param_labels[:event.n_tracks] = param_labels[shuffle_idx]

        # data normalization
        if self.hits_normalizer:
            hits = self.hits_normalizer(hits)

        return {
            "hits": hits,
            "hit_labels": hit_labels,
            "params": params,
            "param_labels": param_labels,
            "mask": np.ones(len(hits), dtype=np.int32)
        }
