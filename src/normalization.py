from typing import Annotated, Literal, Tuple, TypeVar, Union

import gin
import numpy as np
import numpy.typing as npt
import torch

from src.constants import OX_RANGE, OY_RANGE, OZ_RANGE, PHI_RANGE, PT_RANGE, THETA_RANGE
from src.data_generation import TrackParams, Vertex

DType = TypeVar("DType", bound=np.generic)
NormTParamsArr = Annotated[npt.NDArray[DType], Literal[8]]
TParamsArr = Annotated[npt.NDArray[DType], Literal[7]]
ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]


class HitsNormalizer:
    """
    MinMax scaler in range from -1 to 1 for each coordinate
    """

    def __init__(
        self,
        x_coord_range: Tuple[float, float] = OX_RANGE,
        y_coord_range: Tuple[float, float] = OY_RANGE,
        z_coord_range: Tuple[float, float] = OZ_RANGE,
    ) -> None:
        self._x_min, self._x_max = x_coord_range
        self._y_min, self._y_max = y_coord_range
        self._z_min, self._z_max = z_coord_range

    @staticmethod
    def _normalize(
        x: Union[np.float32, npt.NDArray],
        min_val: float,
        max_val: float,
    ) -> Union[np.float32, npt.NDArray]:
        return 2 * (x - min_val) / (max_val - min_val) - 1

    @staticmethod
    def _denormalize(
        x: Union[np.float32, npt.NDArray],
        min_val: float,
        max_val: float,
    ) -> Union[np.float32, npt.NDArray]:
        return (x + 1) * (max_val - min_val) / 2 + min_val

    def normalize(self, inputs: ArrayNx3[np.float32]) -> ArrayNx3[np.float32]:
        x_norm = self._normalize(inputs[:, 0], self._x_min, self._x_max)
        y_norm = self._normalize(inputs[:, 1], self._y_min, self._y_max)
        z_norm = self._normalize(inputs[:, 2], self._z_min, self._z_max)
        norm_inputs = np.hstack(
            [x_norm.reshape(-1, 1), y_norm.reshape(-1, 1), z_norm.reshape(-1, 1)]
        )
        return norm_inputs

    def denormalize(self, inputs: ArrayNx3[np.float32]) -> ArrayNx3[np.float32]:
        x_norm = self._denormalize(inputs[:, 0], self._x_min, self._x_max)
        y_norm = self._denormalize(inputs[:, 1], self._y_min, self._y_max)
        z_norm = self._denormalize(inputs[:, 2], self._z_min, self._z_max)
        denorm_inputs = np.hstack(
            [x_norm.reshape(-1, 1), y_norm.reshape(-1, 1), z_norm.reshape(-1, 1)]
        )
        return denorm_inputs

    def __call__(self, inputs: ArrayNx3[np.float32]) -> ArrayNx3[np.float32]:
        return self.normalize(inputs)


@gin.configurable
class TrackParamsNormalizer:
    """
    Prepares track params vector to be fed to NN loss function by
    normalizing them from 0 to 1 and converting binary parameter
    charge to categorical
    """

    def __init__(
        self,
        vx_range: Tuple[float, float] = OX_RANGE,
        vy_range: Tuple[float, float] = OY_RANGE,
        vz_range: Tuple[float, float] = OZ_RANGE,
        pt_range: Tuple[float, float] = PT_RANGE,
        phi_range: Tuple[float, float] = PHI_RANGE,
        theta_range: Tuple[float, float] = THETA_RANGE,
    ) -> None:
        self._vx_range = vx_range
        self._vy_range = vy_range
        self._vz_range = vz_range
        self._pt_range = pt_range
        self._phi_range = phi_range
        self._theta_range = theta_range

    @staticmethod
    def minmax_norm(p: np.float32, min_val: np.float32, max_val: np.float32):
        return (p - min_val) / (max_val - min_val)

    @staticmethod
    def minmax_denorm(p: np.float32, min_val: np.float32, max_val: np.float32):
        return p * (max_val - min_val) + min_val

    def normalize(
        self,
        vx: np.float32,
        vy: np.float32,
        vz: np.float32,
        pt: np.float32,
        phi: np.float32,
        theta: np.float32,
        charge: np.int8,
    ) -> NormTParamsArr:
        """Normalizes vertex (vx, vy, vz) + params (pt, phi, theta, charge)

        Track params are generated in the following way:
        ```python
            track_params = TrackParams(
                pt=np.random.uniform(100, 1000),  # MeV / c
                phi=np.random.uniform(0, 2*np.pi),
                theta=np.arccos(np.random.uniform(-1, 1)),
                charge=np.random.choice([-1, 1]).astype("int8"),
            )
        ```
        """
        params_list = []
        for data, low, up in (
            (vx, *self._vx_range),
            (vy, *self._vy_range),
            (vz, *self._vz_range),
            (pt, *self._pt_range),
            (phi, *self._phi_range),
            (theta, *self._theta_range),
        ):
            params_list.append(self.minmax_norm(data, low, up))

        params_list.append(0 if charge == -1 else 1)

        return np.array(params_list, dtype=np.float64)

    def denormalize(
        self,
        norm_params_vector: NormTParamsArr,
        is_charge_categorical: bool = True,
        is_numpy: bool = True,
    ) -> TParamsArr:
        """Function that denormalizes normalized track parameters including vertex
        to return to the original values
        """

        denorm_params_list = []
        for data, low, up in (
            (norm_params_vector[0], *self._vx_range),
            (norm_params_vector[1], *self._vy_range),
            (norm_params_vector[2], *self._vz_range),
            (norm_params_vector[3], *self._pt_range),
            (norm_params_vector[4], *self._phi_range),
            (norm_params_vector[5], *self._theta_range),
        ):
            denorm_params_list.append(self.minmax_denorm(data, low, up))

        # from categorical (0, 1) to (-1, 1)
        if is_charge_categorical:
            charge = (norm_params_vector[6] > 0.5) * 2 - 1
        else:
            charge = norm_params_vector[6]

        denorm_params_list.append(charge)

        if is_numpy:
            params_vector = np.array(denorm_params_list, dtype=np.float32)
        else:
            denorm_params_list_tensor = list(
                map(lambda x: torch.tensor(x, dtype=torch.float32), denorm_params_list)
            )
            params_vector = torch.stack(denorm_params_list_tensor)

        return params_vector


if __name__ == "__main__":
    p = TrackParams(
        phi=np.float32(5.9388142319757575),
        theta=np.float32(0.7744413183107033),
        pt=np.float32(946.6066686063032),
        charge=np.int8(-1),
    )
    v = Vertex(
        x=np.float32(-0.705010350151656),
        y=np.float32(-18.174579211083792),
        z=np.float32(44.13365795124486),
    )

    normalizer = TrackParamsNormalizer()
    normalized_params = normalizer.normalize(
        vx=v.x, vy=v.y, vz=v.z, pt=p.pt, phi=p.phi, theta=p.theta, charge=p.charge
    )
    params = normalizer.denormalize(normalized_params, is_numpy=True)

    print(params[0] - v.x, params[1] - v.y, params[2] - v.z)
    print(
        params[3] - p.pt, params[4] - p.phi, params[5] - p.theta, params[6] - p.charge
    )
