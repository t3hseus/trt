from typing import Any, Callable, Tuple

import numpy as np
import torch

from src.constants import OZ_RANGE
from src.data_generation import ArrayNx3, Momentum, Point, TrackParams, Vertex, \
    TorchTrackParams, TorchVertex, TorchPoint, TorchMomentum
from src.normalization import (ConstraintsNormalizer, TParamsArr,
                               TrackParamsNormalizer)

GenFN = Callable[
    [
        TrackParams,
        Vertex,
        np.ndarray[Any, np.float32],
    ],
    Tuple[ArrayNx3[np.float32], ArrayNx3[np.float32], TrackParams],
]


class TracksFromParamsGenerator:
    def __init__(
        self,
        params_normalizer: TrackParamsNormalizer,
        hits_normalizer: ConstraintsNormalizer,
        n_stations: int = 25,
        r_coord_range: tuple[float, float] = (270, 850),
        z_coord_range: tuple[float, float] = OZ_RANGE,
        magnetic_field: float = 0.8,
    ):
        self.hits_normalizer = hits_normalizer
        self.params_normalizer = params_normalizer
        self.radii = np.linspace(r_coord_range[0], r_coord_range[1], n_stations)  # mm
        self.z_coord_range = z_coord_range
        self.magnetic_field = magnetic_field

    def generate_tracks(self, params: TParamsArr):
        tracks = []
        momentums = []
        for i, params_vector in enumerate(params):
            temp_params = self.params_normalizer.denormalize(
                norm_params_vector=params_vector, is_charge_categorical=False,
                is_numpy=False
            )
            params = TorchTrackParams(
                pt=temp_params[3],
                phi=temp_params[4],
                theta=temp_params[5],
                charge=temp_params[6],
            )
            vertex = TorchVertex(*temp_params[:3])
            track, momentum, params = self.generate_track_hits(
                track_params=params, vertex=vertex, radii=self.radii
            )
            tracks.append(track)
            momentums.append(momentum)
        return tracks, momentums

    def generate_track_hits(self, track_params, radii, vertex):
        hits, momentums = [], []
        for _, r in enumerate(radii):
            point, momentum = self.generate_hit(
                track_params=track_params,
                vertex=vertex,
                Rc=r,
                magnetic_field=self.magnetic_field,
            )

            # build hit
            hit = TorchPoint(point.x, point.y, point.z)

            hits.append(hit.torch)
            momentums.append(momentum.torch)

        hits = torch.stack(hits, 0)
        momentums = torch.stack(momentums, 0)
        return hits, momentums, track_params

    @staticmethod
    def generate_hit(
        track_params: TorchTrackParams,
        vertex: TorchVertex,
        Rc: float,
        magnetic_field: float = 0.8,
    ) -> Tuple[TorchPoint, TorchMomentum]:
        """Generates a single hit with its momentum by params"""

        R = track_params.pt / 0.29 / magnetic_field  # mm
        k0 = R / torch.tan(track_params.theta)
        x0 = vertex.x + R * torch.cos(track_params.phit)
        y0 = vertex.y + R * torch.sin(track_params.phit)

        Rtmp = Rc - vertex.r

        if R < Rtmp / 2 or R == 0:  # no intersection
            return TorchPoint(0, 0, 0), TorchMomentum(0, 0, 0)

        R = track_params.charge * R  # both polarities
        alpha = 2 * torch.arcsin(Rtmp / 2 / R)

        if alpha > torch.pi:
            return TorchPoint(0, 0, 0), TorchMomentum(
                0, 0, 0
            )  # algorithm doesn't work for spinning tracks

        extphi = track_params.phi - alpha / 2
        if extphi > (2 * torch.pi):
            extphi = extphi - 2 * torch.pi

        if extphi < 0:
            extphi = extphi + 2 * torch.pi

        x = vertex.x + Rtmp * torch.cos(extphi)
        y = vertex.y + Rtmp * torch.sin(extphi)

        radial = torch.tensor(
            [x - x0 * track_params.charge, y - y0 * track_params.charge],
            dtype=torch.float32,
        )

        rotation_matrix = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
        tangent = torch.matmul(rotation_matrix, radial)

        tangent /= torch.sqrt(tangent.square().sum())  # pt
        tangent *= -track_params.pt * track_params.charge
        px, py = tangent[0], tangent[1]

        z = vertex.z + k0 * alpha

        return TorchPoint(x, y, z), TorchMomentum(px, py, track_params.pz)
