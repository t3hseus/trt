import numpy as np
import dataclasses as _dc
from typing import Tuple, Optional, Any, Mapping


@_dc.dataclass(frozen=True)
class TrackParams:
    phi: np.float32
    theta: np.float32
    pt: np.float32
    charge: np.int32  # -1 or 1
    vz: np.float32


@_dc.dataclass(frozen=True)
class Event:
    """Single generated event"""
    hits: np.ndarray[(Any, 3), np.float32]
    momentums: np.ndarray[(Any, 3), np.float32]  # px, py, pz
    fakes: np.ndarray[(Any, 3), np.float32]  # fake hits
    track_ids: np.ndarray[Any, np.float32]
    missing_hits_mask: np.ndarray[Any, np.bool_]
    # mapping from track_id to its params
    track_params: Mapping[int, TrackParams]

    def __str__(self) -> str:
        event_str = (
            "Event:\n"
            f"Shape of real hits: {self.hits.shape}\n"
            f"Shape of momentums: {self.momentums.shape}\n"
            f"Shape of fake hits: {self.fakes.shape}\n"
            f"Fraction of fakes: {len(self.fakes) / (len(self.hits) + len(self.fakes)):.2f}\n"
            f"Fraction of missing hits: {np.sum(self.missing_hits_mask) / len(self.hits):.2f}\n"
            f"Number of unique tracks: {np.unique(self.track_ids).size}\n"
            "Track parameters:"
        )
        track_params_str = "\n".join([
            (f"\tTrack ID: {tid}, Params: phi={p.phi:.2f}, "
             f"theta={p.theta:.2f}, pt={p.pt:.2f}, charge={p.charge}, "
             f"vz={p.vz:.2f}") for tid, p in self.track_params.items()
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
        pt: float,
        charge: int,
        theta: float,
        phi: float,
        z0: float,
        Rc: float
    ) -> Tuple[float, float, float, float, float, float]:
        # deg = 180 / np.pi
        pz = pt / np.tan(theta) * charge
        phit = phi - np.pi / 2
        R = pt / 0.29 / self.magnetic_field  # mm
        k0 = R / np.tan(theta)
        x0 = R * np.cos(phit)
        y0 = R * np.sin(phit)

        if R < Rc / 2:  # no intersection
            return (0, 0, 0)

        R = charge * R  # both polarities
        alpha = 2 * np.arcsin(Rc / 2 / R)

        if (alpha > np.pi):
            return (0, 0, 0)  # algorithm doesn't work for spinning tracks

        extphi = phi - alpha / 2
        if extphi > (2 * np.pi):
            extphi = extphi - 2 * np.pi

        if extphi < 0:
            extphi = extphi + 2 * np.pi

        x = Rc * np.cos(extphi)
        y = Rc * np.sin(extphi)

        radial = np.array([x - x0*charge, y - y0*charge], dtype=np.float32)

        rotation_matrix = np.array([[0, -1], [1, 0]], dtype=np.float32)
        tangent = np.dot(rotation_matrix, radial)

        tangent /= np.sqrt(np.sum(np.square(tangent)))  # pt
        tangent *= -pt * charge
        px, py = tangent[0], tangent[1]

        z = z0 + k0 * alpha
        return (x, y, z, px, py, pz)

    def generate_track_hits(
        self,
        vx: float,
        vy: float,
        vz: float,
        radii: np.ndarray[Any, np.float32],
        detector_eff: Optional[float] = None
    ) -> Tuple[np.ndarray[(Any, 3), np.float32], np.ndarray[(Any, 3), np.float32], TrackParams]:
        if detector_eff is None:
            detector_eff = self.detector_eff

        hits, momentums = [], []
        pt = np.random.uniform(100, 1000)  # MeV / c
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.arccos(np.random.uniform(-1, 1))
        charge = np.random.choice([-1, 1]).astype("int8")

        for _, r in enumerate(radii):
            x, y, z, px, py, pz = self.extrapolate_to_r(
                pt, charge, theta, phi, vz, r)

            if (x, y, z) == (0, 0, 0):
                continue

            if z >= 2386 or z <= -2386:
                continue

            z = z + np.random.normal(0, 0.1)
            phit = np.arctan2(x, y)
            delta = np.random.normal(0, 0.1)
            x = x + delta * np.sin(phit)
            y = y - delta * np.cos(phit)

            if np.random.uniform(0, 1) < detector_eff:
                hits.append([x, y, z])
                momentums.append([px, py, pz])
            else:
                # add zeros for missing hit
                hits.append([0, 0, 0])
                momentums.append([0, 0, 0])

        hits = np.asarray(hits, dtype=np.float32)
        momentums = np.asarray(momentums, dtype=np.float32)
        track_params = TrackParams(
            phi=phi,
            theta=theta,
            pt=pt,
            charge=charge,
            vz=vz
        )
        return hits, momentums, track_params

    def generate_fakes(
        self,
        n_tracks: int,
        radii: np.ndarray[Any, np.float32]
    ) -> np.ndarray[(Any, 3), np.float32]:
        max_fakes = n_tracks**2 * len(radii)
        min_fakes = max_fakes / 2

        n_fakes = np.random.randint(min_fakes, max_fakes)
        R = np.random.choice(radii, size=n_fakes)
        Phi = np.random.uniform(0, 2*np.pi, size=n_fakes)
        Z = np.random.uniform(*self.z_coord_range, size=n_fakes)
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)

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
        vx = np.random.normal(*self.vx_range)
        vy = np.random.normal(*self.vy_range)
        vz = np.random.uniform(*self.vz_range)
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
                    vx=vx,
                    vy=vy,
                    vz=vz,
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
            momentums=momentums,
            fakes=fakes,
            track_ids=track_ids,
            track_params=params,
            missing_hits_mask=missing_hits_mask
        )


if __name__ == "__main__":
    event_gen = SPDEventGenerator()

    for i in range(10):
        event = event_gen.generate_spd_event()
        print(event)
