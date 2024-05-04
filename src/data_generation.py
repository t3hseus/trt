import numpy as np
import numpy.typing as npt
import dataclasses as _dc
from functools import cached_property
from typing import Tuple, Optional, Any, Mapping
from typing import Annotated, Literal, TypeVar


DType = TypeVar("DType", bound=np.generic)
TParamsArr = Annotated[
    npt.NDArray[DType],
    Literal["phi", "theta", "pt", "charge"]
]
Array3 = Annotated[npt.NDArray[DType], Literal[3]]
ArrayN = Annotated[npt.NDArray[DType], Literal["N"]]
ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]


@_dc.dataclass(frozen=True)
class Momentum:
    px: np.float32
    py: np.float32
    pz: np.float32

    @cached_property
    def numpy(self) -> Array3[np.float32]:
        return np.asarray([self.px, self.py, self.pz], dtype=np.float32)

    def __str__(self) -> str:
        return f"Momentum(px={self.px:.2f}, py={self.py:.2f}, pz={self.pz:.2f})"


@_dc.dataclass(frozen=True)
class Point:
    x: np.float32
    y: np.float32
    z: np.float32

    @cached_property
    def r(self) -> np.float32:
        return np.sqrt(self.x*self.x + self.y*self.y)

    @cached_property
    def numpy(self) -> Array3[np.float32]:
        return np.asarray([self.x, self.y, self.z], dtype=np.float32)

    def __str__(self) -> str:
        return f"Point(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


@_dc.dataclass(frozen=True)
class Vertex(Point):

    def __str__(self) -> str:
        return f"Vertex(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


@_dc.dataclass(frozen=True)
class TrackParams:
    phi: np.float32
    theta: np.float32
    pt: np.float32
    charge: np.int32  # -1 or 1

    @cached_property
    def pz(self) -> np.float32:
        return self.pt / np.tan(self.theta) * self.charge

    @cached_property
    def phit(self) -> np.float32:
        return self.phi - np.pi / 2

    @cached_property
    def numpy(self) -> TParamsArr[np.float32]:
        return np.asarray([self.phi, self.theta, self.pt, self.charge], dtype=np.float32)

    def __str__(self) -> str:
        return (f"TrackParams(phi={self.phi:.2f}, theta={self.theta:.2f}, "
                f"pt={self.pt:.2f}, charge={self.charge})")


class Event:
    """Single generated event"""

    def __init__(
            self,
            hits: ArrayNx3[np.float32],
            track_ids: ArrayN[np.int32],
            momentums: ArrayNx3[np.float32],
            fakes: ArrayNx3[np.float32],
            missing_hits_mask: ArrayN[np.bool_],
            vertex: Vertex,
            track_params: Mapping[int, TrackParams]
    ) -> None:
        # original values before applying missing_hits_mask
        self._hits = hits
        self._track_ids = track_ids
        self._momentums = momentums  # px, py, pz
        self._fakes = fakes  # fake hits
        self._missing_hits_mask = missing_hits_mask
        self._vertex = vertex  # vx, vy, vz
        # mapping from track_id to its params
        self._track_params = track_params

    @cached_property
    def hits(self) -> ArrayNx3[np.float32]:
        """Apply missing hits mask to get the final array of hits"""
        return self._hits[~self._missing_hits_mask]

    @cached_property
    def track_ids(self) -> ArrayN[np.int32]:
        """Apply missing hits mask to get the final array of track ids"""
        return self._track_ids[~self._missing_hits_mask]

    @cached_property
    def momentums(self) -> ArrayNx3[np.float32]:
        """Apply missing hits mask to get the final array of momentums"""
        return self._momentums[~self._missing_hits_mask]

    @property
    def fakes(self) -> ArrayNx3[np.float32]:
        return self._fakes

    @property
    def missing_hits_mask(self) -> ArrayN[np.bool_]:
        return self._missing_hits_mask

    @property
    def vertex(self) -> Vertex:
        return self._vertex

    @property
    def track_params(self) -> Mapping[int, TrackParams]:
        return self._track_params

    @cached_property
    def n_tracks(self) -> np.int32:
        return np.unique(self.track_ids).size

    def __str__(self) -> str:
        event_str = (
            "Event:\n"
            f"Shape of real hits: {self.hits.shape}\n"
            f"Shape of momentums: {self.momentums.shape}\n"
            f"Shape of fake hits: {self.fakes.shape}\n"
            f"Fraction of fakes: {len(self.fakes) / (len(self.hits) + len(self.fakes)):.2f}\n"
            f"Fraction of missing hits: {np.sum(self.missing_hits_mask) / len(self.hits):.2f}\n"
            f"Number of unique tracks: {self.n_tracks}\n"
            f"Vertex: {self.vertex}\n"
            "Track parameters:"
        )
        track_params_str = "\n".join([
            f"\tTrack ID: {tid}, {params}"
            for tid, params in self.track_params.items()
        ])
        return "\n".join([event_str, track_params_str, "\n"])


class SPDEventGenerator:
    def __init__(
        self,
        max_event_tracks: int = 10,
        detector_eff: float = 1.0,
        add_fakes: bool = True,
        n_stations: int = 35,
        vx_range: Tuple[float, float] = (0.0, 10.0),
        vy_range: Tuple[float, float] = (0.0, 10.0),
        vz_range: Tuple[float, float] = (-300.0, 300.),
        z_coord_range: Tuple[float, float] = (-2386.0, 2386.0),
        r_coord_range: Tuple[float, float] = (270, 850),
        magnetic_field: float = 0.8  # magnetic field [T]
    ):
        self.max_event_tracks = max_event_tracks
        self.detector_eff = detector_eff
        self.add_fakes = add_fakes
        self.n_stations = n_stations
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vz_range = vz_range
        self.z_coord_range = z_coord_range
        self.r_coord_range = r_coord_range
        self.magnetic_field = magnetic_field

    def extrapolate_to_r(
        self,
        track_params: TrackParams,
        vertex: Vertex,
        Rc: float
    ) -> Tuple[Point, Momentum]:

        R = track_params.pt / 0.29 / self.magnetic_field  # mm
        k0 = R / np.tan(track_params.theta)
        x0 = vertex.x + R * np.cos(track_params.phit)
        y0 = vertex.y + R * np.sin(track_params.phit)

        Rtmp = Rc - vertex.r

        if R < Rtmp / 2:  # no intersection
            return (0, 0, 0)

        R = track_params.charge * R  # both polarities
        alpha = 2 * np.arcsin(Rtmp / 2 / R)

        if (alpha > np.pi):
            return (0, 0, 0)  # algorithm doesn't work for spinning tracks

        extphi = track_params.phi - alpha / 2
        if extphi > (2 * np.pi):
            extphi = extphi - 2 * np.pi

        if extphi < 0:
            extphi = extphi + 2 * np.pi

        x = vertex.x + Rtmp * np.cos(extphi)
        y = vertex.y + Rtmp * np.sin(extphi)

        radial = np.array([
            x - x0*track_params.charge,
            y - y0*track_params.charge
        ], dtype=np.float32)

        rotation_matrix = np.array([[0, -1], [1, 0]], dtype=np.float32)
        tangent = np.dot(rotation_matrix, radial)

        tangent /= np.sqrt(np.sum(np.square(tangent)))  # pt
        tangent *= -track_params.pt * track_params.charge
        px, py = tangent[0], tangent[1]

        z = vertex.z + k0 * alpha
        return (
            Point(x, y, z),
            Momentum(px, py, track_params.pz)
        )

    def generate_track_hits(
        self,
        vertex: Vertex,
        radii: np.ndarray[Any, np.float32],
        detector_eff: Optional[float] = None
    ) -> Tuple[np.ndarray[(Any, 3), np.float32], np.ndarray[(Any, 3), np.float32], TrackParams]:

        if detector_eff is None:
            detector_eff = self.detector_eff

        hits, momentums = [], []
        track_params = TrackParams(
            pt=np.random.uniform(100, 1000),  # MeV / c
            phi=np.random.uniform(0, 2*np.pi),
            theta=np.arccos(np.random.uniform(-1, 1)),
            charge=np.random.choice([-1, 1]).astype("int8"),
        )

        for _, r in enumerate(radii):
            point, momentum = self.extrapolate_to_r(
                track_params=track_params,
                vertex=vertex,
                Rc=r,
            )

            if (point.x, point.y, point.z) == (0, 0, 0):
                continue

            if point.z >= 2386 or point.z <= -2386:  # some magic number
                continue

            z = point.z + np.random.normal(0, 0.1)
            phit = np.arctan2(point.x, point.y)
            delta = np.random.normal(0, 0.1)
            x = point.x + delta * np.sin(phit)
            y = point.y - delta * np.cos(phit)

            # build hit
            hit = Point(x, y, z)

            if np.random.uniform(0, 1) < detector_eff:
                hits.append(hit.numpy)
                momentums.append(momentum.numpy)
            else:
                # add zeros for missing hit
                hits.append(Point(0, 0, 0).numpy)
                momentums.append(Momentum(0, 0, 0).numpy)

        hits = np.asarray(hits, dtype=np.float32)
        momentums = np.asarray(momentums, dtype=np.float32)
        return hits, momentums, track_params

    def generate_fakes(
        self,
        n_tracks: int,
        radii: ArrayN[np.float32]
    ) -> ArrayNx3[np.float32]:
        max_fakes = n_tracks**2 * len(radii)
        min_fakes = max_fakes / 2

        n_fakes = np.random.randint(min_fakes, max_fakes)
        R = np.random.choice(radii, size=n_fakes)
        phi = np.random.uniform(0, 2*np.pi, size=n_fakes)
        Z = np.random.uniform(*self.z_coord_range, size=n_fakes)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)

        fakes = np.column_stack([X, Y, Z])
        return fakes

    def generate_spd_event(
        self,
        detector_eff: Optional[float] = None,
        add_fakes: Optional[bool] = None,
    ) -> Event:
        if detector_eff is None:
            detector_eff = self.detector_eff
        if add_fakes is None:
            add_fakes = self.add_fakes

        radii = np.linspace(
            self.r_coord_range[0],
            self.r_coord_range[1],
            self.n_stations
        )  # mm
        vertex = Vertex(
            x=np.random.normal(*self.vx_range),
            y=np.random.normal(*self.vy_range),
            z=np.random.uniform(*self.vz_range),
        )
        n_tracks = np.random.randint(1, self.max_event_tracks)

        hits = []
        momentums = []
        track_ids = []
        params = {}
        fakes = None

        for track in range(0, n_tracks):
            track_hits = np.asarray([], dtype=np.float32)  # empty array
            # if generator returns empty track, call it again
            # until the needed track will be generated
            while track_hits.size == 0:
                track_hits, track_momentums, track_params = self.generate_track_hits(
                    vertex=vertex,
                    radii=radii,
                    detector_eff=detector_eff
                )
            # add to the global list of hits
            hits.append(track_hits)
            momentums.append(track_momentums)
            params[track] = track_params
            track_ids.append(np.full(len(track_hits), track))

        hits = np.vstack(hits)
        missing_hits_mask = ~hits.any(axis=1)
        momentums = np.vstack(momentums)
        track_ids = np.concatenate(track_ids)

        if add_fakes:
            fakes = self.generate_fakes(
                n_tracks=n_tracks,
                radii=radii
            )

        return Event(
            hits=hits,
            track_ids=track_ids,
            momentums=momentums,
            fakes=fakes,
            track_params=params,
            missing_hits_mask=missing_hits_mask,
            vertex=vertex,
        )


if __name__ == "__main__":
    event_gen = SPDEventGenerator()

    for _ in range(10):
        event = event_gen.generate_spd_event()
        print(event)
