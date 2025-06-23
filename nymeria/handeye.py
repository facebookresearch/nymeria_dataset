# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from loguru import logger
from projectaria_tools.core.sophus import SE3


class HandEyeSolver:
    def __init__(self, smooth: bool, window: int, skip: int = 240, stride: int = 1):
        self.stride = int(stride)
        self.smooth = smooth
        self.skip = int(skip)
        self.window = int(window)
        if self.window < 240:
            self.smooth = False

    def so3xR3(self, T_Wa_A: list[SE3], T_Wb_B: list[SE3]) -> SE3:
        """
        \return T_A_B using so3xR3 SVD decomposition.
        """
        assert len(T_Wa_A) == len(T_Wb_B)

        N = len(T_Wa_A) - self.stride
        se3_A1_A2 = [T_Wa_A[i].inverse() @ T_Wa_A[i + self.stride] for i in range(N)]
        se3_B1_B2 = [T_Wb_B[i].inverse() @ T_Wb_B[i + self.stride] for i in range(N)]

        # solve for R
        log_A1_A2 = [x.rotation().log() for x in se3_A1_A2]
        log_B1_B2 = [x.rotation().log() for x in se3_B1_B2]
        A = np.stack(log_A1_A2, axis=-1).squeeze()
        B = np.stack(log_B1_B2, axis=-1).squeeze()
        logger.debug(f"{A.shape=}, {B.shape=}")

        matrixU, S, matrixVh = np.linalg.svd(
            B @ A.transpose(), full_matrices=True, compute_uv=True
        )
        logger.debug(f"{matrixU.shape=}, {S.shape=}, {matrixVh.shape=}")

        RX = matrixVh.transpose() @ matrixU.transpose()
        if np.linalg.det(RX) < 0:
            RX[2, :] = RX[2, :] * -1.0

        # solve for t
        jacobian = [x.rotation().to_matrix() - np.eye(3) for x in se3_A1_A2]
        jacobian = np.concatenate(jacobian, axis=0)
        assert jacobian.shape == (N * 3, 3)
        logger.debug(f"{jacobian.shape=}")
        residual = [
            RX @ b.translation().reshape(3, 1) - a.translation().reshape(3, 1)
            for a, b in zip(se3_A1_A2, se3_B1_B2)
        ]
        residual = np.concatenate(residual, axis=0)
        assert residual.shape == (N * 3, 1)
        logger.debug(f"{residual.shape=}")
        JTJ = jacobian.T @ jacobian
        JTr = jacobian.T @ residual
        tX = np.linalg.lstsq(JTJ, JTr, rcond=None)[0]

        T_A_B = np.ndarray([3, 4])
        T_A_B[:3, :3] = RX
        T_A_B[:3, 3] = tX.squeeze()
        logger.debug(f"{T_A_B=}\n")
        T_A_B = SE3.from_matrix3x4(T_A_B)
        return T_A_B

    def __call__(self, T_Wa_A: list[SE3], T_Wb_B: list[SE3]) -> list[SE3]:
        N = len(T_Wa_A)
        assert N == len(T_Wb_B)
        if self.window >= N or not self.smooth:
            T_A_B = self.so3xR3(T_Wa_A, T_Wb_B)
            return [T_A_B]

        Ts_A_B = []
        for i in range(0, N, self.skip):
            istart = int(i - self.window / 2)
            if istart < 0:
                istart = 0
            iend = istart + self.window
            if iend >= N:
                iend = -1
                istart = N - self.window

            t_wa_a = T_Wa_A[istart:iend]
            t_wb_b = T_Wb_B[istart:iend]
            T_A_B = self.so3xR3(t_wa_a, t_wb_b)
            Ts_A_B.append(T_A_B)
        return Ts_A_B
