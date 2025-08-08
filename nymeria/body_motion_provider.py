# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pymomentum as pym
import torch
from loguru import logger
from nymeria.xsens_constants import XSensConstants
from projectaria_tools.core.sophus import SE3
from pymomentum.geometry import Character, Mesh


class BodyDataProvider:
    _dt_nominal: float = 1.0e6 / 240.0
    _dt_tolerance: int = 1000  # 1ms
    _tcorrect_tolerance: int = 10_000  # 10ms

    # coordinates transform between momentum and xsens
    _A_Wx_Wm = torch.tensor([0.01, 0, 0, 0, 0, -0.01, 0, 0.01, 0]).reshape([3, 3])

    def __init__(self, npzfile: str, glbfile: str) -> None:
        if not Path(npzfile).is_file():
            logger.error(f"{npzfile=} not found")
            return

        logger.info(f"loading xsens from {npzfile=}")
        self.xsens_data: dict[str, np.ndarray] = dict(np.load(npzfile))
        for k, v in self.xsens_data.items():
            logger.info(f"{k=}, {v.shape=}")

        self.__correct_timestamps()
        self.__correct_quaternion()

        # load glb if exist
        self.character: Character = None
        self.motion: np.ndarray = None
        if Path(glbfile).is_file():
            self.character, self.motion, _, fps = Character.load_gltf_with_motion(
                glbfile
            )
            assert fps == self.xsens_data[XSensConstants.k_framerate]
            assert self.motion.shape[0] == self.xsens_data[XSensConstants.k_frame_count]
            assert self.character.has_mesh

    @property
    def momentum_template_mesh(self) -> Mesh | None:
        if self.character is not None:
            return self.character.mesh
        else:
            return None

    def __correct_timestamps(self) -> None:
        t_original = self.xsens_data[XSensConstants.k_timestamps_us]
        dt_original = t_original[1:] - t_original[:-1]
        invalid = np.abs(dt_original - self._dt_nominal) > self._dt_tolerance
        num_invalid = invalid.sum()
        percentage = num_invalid / t_original.size * 100.0
        if num_invalid == 0:
            return
        logger.warning(f"number of invalid timestamps {num_invalid}, {percentage=}%")
        dt_corrected = dt_original
        dt_corrected[invalid] = int(self._dt_nominal)
        dt_corrected = np.insert(dt_corrected, 0, 0)
        t_corrected = t_original[0] + np.cumsum(dt_corrected)

        t_diff = np.abs(t_corrected - t_original)
        logger.info(f"after correct {t_diff[-1]= }us")
        if t_diff[-1] > self._tcorrect_tolerance:
            raise RuntimeError(f"corrected timestamps exceed tolerance {t_diff[-1]=}")

        self.xsens_data[XSensConstants.k_timestamps_us] = t_corrected

    def __correct_quaternion(self) -> None:
        qWXYZ = self.xsens_data[XSensConstants.k_part_qWXYZ].reshape(
            -1, XSensConstants.num_parts, 4
        )
        qn = np.linalg.norm(qWXYZ, axis=-1, keepdims=False)
        invalid = qn < 0.1
        if invalid.sum() == 0:
            return
        else:
            logger.error(f"number of invalid quaternions {invalid.sum()}")

        for p in range(XSensConstants.num_parts):
            if qn[0, p] < 0.5:
                qWXYZ[0, p] = np.array([1, 0, 0, 0])
        for f in range(1, qn.shape[0]):
            for p in range(XSensConstants.num_parts):
                if qn[f, p] < 0.5:
                    qWXYZ[f, p] = qWXYZ[f - 1, p]
        self.xsens_data[XSensConstants.k_part_qWXYZ] = qWXYZ.reshape(
            -1, XSensConstants.num_parts * 4
        )

    def get_global_timespan_us(self) -> tuple[int, int]:
        t_us = self.xsens_data[XSensConstants.k_timestamps_us]
        return t_us[0], t_us[-1]

    def get_T_w_h(self, timespan_ns: tuple[int, int] = None) -> tuple[list, list]:
        head_idx = XSensConstants.part_names.index("Head")
        num_parts = XSensConstants.num_parts
        timestamps_ns = self.xsens_data[XSensConstants.k_timestamps_us] * 1e3
        if timespan_ns is not None:
            t_start, t_end = timespan_ns
            i_start = np.searchsorted(timestamps_ns, t_start) + 240
            i_end = np.searchsorted(timestamps_ns, t_end) - 240
            assert i_start < i_end
        else:
            i_start = 0
            i_end = None

        head_q = self.xsens_data[XSensConstants.k_part_qWXYZ].reshape(-1, num_parts, 4)[
            i_start:i_end, head_idx, :
        ]
        head_t = self.xsens_data[XSensConstants.k_part_tXYZ].reshape(-1, num_parts, 3)[
            i_start:i_end, head_idx, :
        ]
        T_w_h: list[SE3] = SE3.from_quat_and_translation(
            head_q[:, 0], head_q[:, 1:], head_t
        )
        t_ns: list[int] = timestamps_ns[i_start:i_end].tolist()
        logger.info(f"get {len(T_w_h)} samples for computing alignment")
        return T_w_h, t_ns

    def __get_closest_timestamp_idx(self, t_us: int) -> int:
        if t_us <= self.get_global_timespan_us()[0]:
            return 0
        if t_us >= self.get_global_timespan_us()[-1]:
            return -1

        timestamps = self.xsens_data[XSensConstants.k_timestamps_us]
        idx_rr = np.searchsorted(timestamps, t_us)
        idx_ll = idx_rr - 1
        if abs(timestamps[idx_ll] - t_us) < abs(timestamps[idx_rr] - t_us):
            return idx_ll
        else:
            return idx_rr

    def get_posed_skeleton_and_skin(
        self, t_us: int, T_W_Hx: SE3 = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        \brief Given a query timestamp, return the closest body motion.
        \arg t_us: query timestamp in microsecond.
        \arg T_W_Hx: optional SE3 alignment from XSens head to world coordinates,
             computed from XSens to Aria world coordinates alignment.
        \return First element is XSens posed skeleton. Second element is posed vertices
                of momentum mesh if the momentum retargetted results are loaded.
                We only return the posed mesh vertices, since the triangles and normals stay the same.
        """
        # find closest timestamp
        idx: int = self.__get_closest_timestamp_idx(t_us)

        # get XSens posed skeleton
        q = self.xsens_data[XSensConstants.k_part_qWXYZ][idx]
        t = self.xsens_data[XSensConstants.k_part_tXYZ][idx]
        T_Wx_Px = BodyDataProvider.qt_to_se3(q, t)
        T_W_Wx: SE3 = None
        if T_W_Hx is not None:
            head_idx = XSensConstants.part_names.index("Head")
            T_Hx_Wx = T_Wx_Px[head_idx].inverse()
            T_W_Wx = T_W_Hx @ T_Hx_Wx
            T_W_Px = [T_W_Wx @ T_wx_px for T_wx_px in T_Wx_Px]
            skel_xsens = BodyDataProvider.se3_to_skeleton(T_W_Px)
        else:
            skel_xsens = BodyDataProvider.se3_to_skeleton(T_Wx_Px)

        # get Momentum posed mesh vertices
        if self.character is not None:
            motion = torch.tensor(self.motion[idx])
            skel_state: torch.Tensor = pym.geometry.model_parameters_to_skeleton_state(
                self.character, motion
            )
            skin_momentum: torch.Tensor = self.character.skin_points(skel_state)

            if T_W_Wx is not None:
                t_W_Wx = (
                    torch.tensor(T_W_Wx.translation()).to(torch.float32).reshape([3, 1])
                )
                R_W_Wx = torch.tensor(T_W_Wx.rotation().to_matrix()).to(torch.float32)

                R_W_Wm = R_W_Wx @ self._A_Wx_Wm
                skin_momentum = (R_W_Wm @ skin_momentum.T + t_W_Wx).T
            else:
                skin_momentum = (self._A_Wx_Wm @ skin_momentum.T).T

        return skel_xsens, skin_momentum

    @staticmethod
    def qt_to_se3(part_qWXYZ: np.ndarray, part_tXYZ: np.ndarray) -> list[SE3]:
        """
        \brief Helper function to convert a frame of skeleton representation from
               list of quaternion + translation to SE3.
        """
        q_WXYZ = part_qWXYZ.reshape(XSensConstants.num_parts, 4)
        t_XYZ = part_tXYZ.reshape(XSensConstants.num_parts, 3)
        return SE3.from_quat_and_translation(q_WXYZ[:, 0], q_WXYZ[:, 1:], t_XYZ)

    @staticmethod
    def se3_to_skeleton(part_se3: list[SE3]) -> np.ndarray:
        """
        \brief Helper function to convert a frame of skeleton parameters to 3D wireframe
               for visualization purposes.
        """
        assert len(part_se3) == XSensConstants.num_parts
        children = np.concatenate([b.translation() for b in part_se3[1:]], axis=0)
        parents = np.concatenate(
            [part_se3[p].translation() for p in XSensConstants.kintree_parents[1:]],
            axis=0,
        )
        skeleton_cp = np.stack([children, parents], axis=1)
        assert skeleton_cp.shape == (XSensConstants.num_bones, 2, 3)
        return skeleton_cp.astype(np.float32)


def create_body_data_provider(
    xdata_npz: str, xdata_glb: str
) -> BodyDataProvider | None:
    if Path(xdata_npz).is_file():
        return BodyDataProvider(npzfile=xdata_npz, glbfile=xdata_glb)
    else:
        return None
