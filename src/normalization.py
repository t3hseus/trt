from typing import Annotated, Literal, Tuple, TypeVar, Union

import gin
import numpy as np
import numpy.typing as npt
import torch

from .constants import (OX_RANGE, OY_RANGE, OZ_RANGE, PHI_RANGE, PT_RANGE,
                        THETA_RANGE)

DType = TypeVar("DType", bound=np.generic)
NormTParamsArr = Annotated[npt.NDArray[DType], Literal[8]]
TParamsArr = Annotated[npt.NDArray[DType], Literal[7]]
ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", 3]]


@gin.configurable
class ConstraintsNormalizer:
    """MinMax scaler in range from -1 to 1
    for each coordinate
    """

    def __init__(
        self,
        x_coord_range: Tuple[float, float] = OX_RANGE,
        y_coord_range: Tuple[float, float] = OY_RANGE,
        z_coord_range: Tuple[float, float] = OZ_RANGE,
    ):
        self._x_min, self._x_max = x_coord_range
        self._y_min, self._y_max = y_coord_range
        self._z_min, self._z_max = z_coord_range

    @staticmethod
    def normalize(
        x: Union[np.float32, np.ndarray],
        min_val: float,
        max_val: float,
    ) -> Union[np.float32, np.ndarray]:
        return 2 * (x - min_val) / (max_val - min_val) - 1

    def __call__(self, inputs: ArrayNx3[np.float32]) -> ArrayNx3[np.float32]:
        x_norm = self.normalize(inputs[:, 0], self._x_min, self._x_max)
        y_norm = self.normalize(inputs[:, 1], self._y_min, self._y_max)
        z_norm = self.normalize(inputs[:, 2], self._z_min, self._z_max)
        norm_inputs = np.hstack(
            [x_norm.reshape(-1, 1), y_norm.reshape(-1, 1), z_norm.reshape(-1, 1)]
        )
        return norm_inputs


@gin.configurable
class TrackParamsNormalizer:
    """Prepares track params vector to be fed to NN loss function by
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
        """Function that normalizes vertex (vx, vy, vz) + params (pt, phi, theta, charge)

        Track params are generated in the following way:
        ```
            track_params = TrackParams(
                pt=np.random.uniform(100, 1000),  # MeV / c
                phi=np.random.uniform(0, 2*np.pi),
                theta=np.arccos(np.random.uniform(-1, 1)),
                charge=np.random.choice([-1, 1]).astype("int8"),
            )
        ```
        """
        norm_vx = self.minmax_norm(vx, *self._vx_range)
        norm_vy = self.minmax_norm(vy, *self._vy_range)
        norm_vz = self.minmax_norm(vz, *self._vz_range)
        norm_pt = self.minmax_norm(pt, *self._pt_range)
        norm_phi = self.minmax_norm(phi, *self._phi_range)
        norm_theta = self.minmax_norm(theta, 0, np.pi)
        cat_charge = 0 if charge == -1 else 1
        params_vector = np.array(
            [norm_vx, norm_vy, norm_vz, norm_pt, norm_phi, norm_theta, cat_charge],
            dtype=np.float32,
        )
        return params_vector

    def denormalize(
        self,
        norm_params_vector: NormTParamsArr,
        is_charge_categorical: bool = True,
        is_numpy: bool = True
        # vx: np.float32,
        # vy: np.float32,
        # vz: np.float32,
        # pt: np.float32,
        # phi: np.float32,
        # theta: np.float32,
        # charge: np.int8
    ) -> TParamsArr:
        """Function that denormalizes normalized track parameters including vertex
        to return to the original values
        """
        orig_vx = self.minmax_denorm(norm_params_vector[0], *self._vx_range)
        orig_vy = self.minmax_denorm(norm_params_vector[1], *self._vy_range)
        orig_vz = self.minmax_denorm(norm_params_vector[2], *self._vz_range)
        orig_pt = self.minmax_denorm(norm_params_vector[3], *self._pt_range)
        orig_phi = self.minmax_denorm(norm_params_vector[4], *self._phi_range)
        pi = np.pi if is_numpy else torch.pi
        orig_theta = self.minmax_denorm(norm_params_vector[5], 0, pi)
        # from categorical (0, 1) to (-1, 1)
        if is_charge_categorical:
            orig_charge = norm_params_vector[6] * 2 - 1
        else:
            orig_charge = norm_params_vector[6]
        if is_numpy:
            params_vector = np.array(
                [orig_vx, orig_vy, orig_vz, orig_pt, orig_phi, orig_theta, orig_charge],
                dtype=np.float32,
            )
        else:
            params_vector = torch.stack(
                [orig_vx, orig_vy, orig_vz, orig_pt, orig_phi, orig_theta, orig_charge]
            )
        return params_vector
