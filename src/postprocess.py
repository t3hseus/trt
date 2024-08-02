from typing import Any, Callable, Tuple

import numpy as np
import torch

from src.constants import OZ_RANGE
from src.data_generation import ArrayNx3, Momentum, Point, TrackParams, Vertex
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
                norm_params_vector=params_vector, is_charge_categorical=False
            )
            params = TrackParams(
                pt=temp_params[3],
                phi=temp_params[4],
                theta=temp_params[5],
                charge=temp_params[6],
            )
            vertex = Vertex(*temp_params[:3])
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
            hit = Point(point.x, point.y, point.z)

            hits.append(hit.numpy)
            momentums.append(momentum.numpy)

        hits = np.asarray(hits, dtype=np.float32)
        momentums = np.asarray(momentums, dtype=np.float32)
        return hits, momentums, track_params

    @staticmethod
    def generate_hit(
        track_params: TrackParams,
        vertex: Vertex,
        Rc: float,
        magnetic_field: float = 0.8,
    ) -> Tuple[Point, Momentum]:
        """Generates a single hit with its momentum by params"""

        R = track_params.pt / 0.29 / magnetic_field  # mm
        k0 = R / np.tan(track_params.theta)
        x0 = vertex.x + R * np.cos(track_params.phit)
        y0 = vertex.y + R * np.sin(track_params.phit)

        Rtmp = Rc - vertex.r

        if R < Rtmp / 2:  # no intersection
            return Point(0, 0, 0), Momentum(0, 0, 0)

        R = track_params.charge * R  # both polarities
        alpha = 2 * np.arcsin(Rtmp / 2 / R)

        if alpha > np.pi:
            return Point(0, 0, 0), Momentum(
                0, 0, 0
            )  # algorithm doesn't work for spinning tracks

        extphi = track_params.phi - alpha / 2
        if extphi > (2 * np.pi):
            extphi = extphi - 2 * np.pi

        if extphi < 0:
            extphi = extphi + 2 * np.pi

        x = vertex.x + Rtmp * np.cos(extphi)
        y = vertex.y + Rtmp * np.sin(extphi)

        radial = np.array(
            [x - x0 * track_params.charge, y - y0 * track_params.charge],
            dtype=np.float32,
        )

        rotation_matrix = np.array([[0, -1], [1, 0]], dtype=np.float32)
        tangent = np.dot(rotation_matrix, radial)

        tangent /= np.sqrt(np.sum(np.square(tangent)))  # pt
        tangent *= -track_params.pt * track_params.charge
        px, py = tangent[0], tangent[1]

        z = vertex.z + k0 * alpha

        return (Point(x, y, z), Momentum(px, py, track_params.pz))
