import numpy as np
from enum import IntEnum
from typing import Optional, Callable, Dict
from torch.utils.data import Dataset
from .data_generation import SPDEventGenerator


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
        hits_normalizer: Optional[Callable] = None,
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
        params = np.zeros((self._max_event_tracks, 7), dtype=np.float32)

        param_labels = np.full(self._max_event_tracks,
                               self._padding_label, dtype=np.int32)
        for i, (track_id, track_params) in enumerate(event.track_params.items()):
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

        return {
            "hits": hits,
            "hit_labels": hit_labels,
            "params": params,
            "param_labels": param_labels,
            "mask": np.ones(len(hits), dtype=np.int32)
        }
