from typing import Tuple, Literal, Optional, Union

import numpy as np
import torch

from src.data_generation import TrackParams, Vertex, SPDEventGenerator
from src.normalization import TParamsArr, TrackParamsNormalizer

# typing
Hits = np.ndarray[Literal["N", 3], np.dtype[np.float32]]
TrackIDs = np.ndarray[Literal["N", ], np.dtype[np.int32]]


class EventRecoveryFromPredictions:
    def __init__(
        self,
        event_generator: SPDEventGenerator,
        track_params_normalizer: Optional[TrackParamsNormalizer] = None,
    ):
        self._event_gen = event_generator
        self._track_params_normalizer = track_params_normalizer

    def _predictions_to_params(
        self,
        pred_params: torch.Tensor,
        pred_charges: torch.Tensor,
        charges_from_categorical: bool = True,
    ) -> TParamsArr:
        charges = pred_charges
        if charges_from_categorical:
            # TODO: unify this to a single format for outputs and predictions
            charges = torch.argmax(pred_charges, dim=-1)
        charges = charges.to(torch.float32) * 2 - 1
        charges = charges.unsqueeze(-1)
        params = torch.concat((pred_params, charges), dim=-1)
        params = params.detach().cpu().numpy()
        return params

    def __call__(
            self,
            # TODO: use one argument instead of two
            pred_params: torch.Tensor,
            pred_charges: torch.Tensor,
            from_targets: bool = False,
            group_by_tracks: bool = False
    ) -> Union[Tuple[Hits, TrackIDs], list[Hits]]:
        """Generates event hits for chosen prediction.
        """
        # TODO: currently supports only tensors with shape [N, P], without batch, fix it!
        if len(pred_params.shape) != 2:
            raise ValueError(
                f"Only two dimensional tensors are supported, squeeze batch dimension {pred_params.shape}!"
            )

        # TODO: use tensors instead of numpy
        track_params = self._predictions_to_params(
            pred_params=pred_params,
            pred_charges=pred_charges,
            charges_from_categorical=not from_targets
        )

        # denormalization
        if self._track_params_normalizer:
            track_params = np.apply_along_axis(
                self._track_params_normalizer.denormalize, axis=1, arr=track_params)

        event_hits = []
        track_ids = []
        for track, param_vector in enumerate(track_params):
            track_hits = self._event_gen.reconstruct_track_hits_from_params(
                track_params=TrackParams(
                    pt=param_vector[3],
                    phi=param_vector[4],
                    theta=param_vector[5],
                    charge=param_vector[6],
                ),
                vertex=Vertex(*param_vector[:3])
            )
            event_hits.append(track_hits)
            track_ids.append(np.full(len(track_hits), track))

        event_hits = np.vstack(event_hits)
        track_ids = np.concatenate(track_ids)

        if group_by_tracks:
            tracks = np.split(
                event_hits,
                np.cumsum(np.unique(track_ids, return_counts=True)[1])
            )[:-1]
            return tracks
        # else
        return event_hits, track_ids
