# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from nymeria.body_motion_provider import BodyDataProvider, create_body_data_provider
from nymeria.definitions import BodyModel
from nymeria.handeye import HandEyeSolver
from nymeria.mhr_provider import create_mhr_data_provider, MhrDataProvider

try:
    import mhr as _mhr  # noqa: F401

    _HAS_MHR = True
except ImportError:
    _HAS_MHR = False
from nymeria.path_provider import SequencePathProvider
from nymeria.recording_data_provider import (
    create_recording_data_provider,
    RecordingDataProvider,
)

try:
    import smplx as _smplx  # noqa: F401

    _HAS_SMPLX = True
except ImportError:
    _HAS_SMPLX = False
from nymeria.smpl_provider import create_smpl_data_provider, SmplDataProvider
from projectaria_tools.core.mps import ClosedLoopTrajectoryPose
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.sophus import SE3


@dataclass(frozen=True)
class NymeriaDataProviderConfig:
    sequence_rootdir: Path
    load_head: bool = True
    load_observer: bool = True
    load_wrist: bool = True
    body_model: Union[BodyModel, None] = BodyModel.MOMENTUM

    # Path to the SMPL model .pkl file (required when body_model == BodyModel.SMPL).
    smpl_model_path: Union[str, None] = None

    # If true, the filtered semidense points are exported into a npz file at the first loading
    view_cached_points: bool = True
    # Parameters for filtering semidense points
    th_invdep: float = 0.0004
    th_dep: float = 0.02
    max_point_count: int = 100_000

    trajectory_sample_fps: float = 1

    # Parameters for solving XSens to Aria world coordinates alignment
    handeye_smooth: bool = False
    handeye_window: int = 240 * 120
    handeye_skip: int = 240 * 5
    handeye_stride: int = 2


class NymeriaDataProvider(NymeriaDataProviderConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        seq_pd = SequencePathProvider(self.sequence_rootdir)

        # create data provider for Aria recordings and MPS output
        self.recording_head = (
            create_recording_data_provider(seq_pd.recording_head)
            if self.load_head
            else None
        )
        self.recording_lwrist = (
            create_recording_data_provider(seq_pd.recording_lwrist)
            if self.load_wrist
            else None
        )
        self.recording_rwrist = (
            create_recording_data_provider(seq_pd.recording_rwrist)
            if self.load_wrist
            else None
        )
        self.recording_observer = (
            create_recording_data_provider(seq_pd.recording_observer)
            if self.load_observer
            else None
        )

        # create data provider for body motion
        self.body_dp: Union[
            BodyDataProvider, SmplDataProvider, MhrDataProvider, None
        ] = None
        if self.body_model == BodyModel.MOMENTUM:
            self.body_dp = create_body_data_provider(
                xdata_npz=seq_pd.body_paths.xsens_processed,
                xdata_glb=seq_pd.body_paths.momentum_model,
            )
        elif self.body_model == BodyModel.SMPL:
            if not _HAS_SMPLX:
                raise ImportError("smplx package is required for SMPL body model. ")
            if self.smpl_model_path is None:
                raise ValueError(
                    "smpl_model_path must be set when body_model is BodyModel.SMPL"
                )
            self.body_dp = create_smpl_data_provider(
                npzfile=seq_pd.smpl_paths.smpl_neutral,
                model_path=self.smpl_model_path,
            )
            if self.body_dp is None:
                raise RuntimeError(
                    "Failed to load SMPL body model. "
                    f"Check that {seq_pd.smpl_paths.smpl_neutral} exists."
                )
        elif self.body_model == BodyModel.MHR:
            if not _HAS_MHR:
                raise ImportError("mhr package is required for MHR body model. ")
            self.body_dp = create_mhr_data_provider(
                glbfile=seq_pd.mhr_paths.mhr_model,
            )
            if self.body_dp is None:
                raise RuntimeError(
                    "Failed to load MHR body model. "
                    f"Check that {seq_pd.mhr_paths.mhr_model} exists."
                )
        # else: body_model is None, body_dp stays None

        if self.body_dp is None and len(self.get_existing_recordings()) == 0:
            raise RuntimeError(
                "data provider is empty. "
                "Make sure there is at least 1 recording or body motion"
            )

        # get overlapping timeline
        self.timespan_ns: tuple[int, int] = self.__get_timespan_ns()

        # compute xsens to aria world alignment (only for MOMENTUM)
        self.__compute_xsens_to_aria_alignment()

    def get_existing_recordings(self) -> list[RecordingDataProvider]:
        return [
            x
            for x in [
                self.recording_head,
                self.recording_observer,
                self.recording_lwrist,
                self.recording_rwrist,
            ]
            if x is not None
        ]

    def __get_timespan_ns(self, ignore_ns: int = 1e9) -> tuple[int, int]:
        """
        \brief Compute overlapping timeline across all loaded data
        """
        t_start = 0
        t_end = None

        if self.body_dp is not None and not isinstance(self.body_dp, MhrDataProvider):
            t0, t1 = self.body_dp.get_global_timespan_us()
            t_start = t0 * 1e3
            t_end = t1 * 1e3

        for rec in self.get_existing_recordings():
            t0, t1 = rec.get_global_timespan_ns()
            t_start = t_start if t_start > t0 else t0
            t_end = t_end if t_end is not None and t_end < t1 else t1

        t_start += ignore_ns
        t_end -= ignore_ns
        assert t_start < t_end, f"invalid time span {t_start= }us, {t_end= }us"

        t_start = int(t_start)
        t_end = int(t_end)
        duration = (t_end - t_start) / 1.0e9
        logger.info(f"time span: {t_start= }us {t_end= }us {duration= }s")
        return t_start, t_end

    def get_synced_rgb_videos(self, t_ns_global: int) -> dict[str, any]:
        data = {}
        for rec in [self.recording_head, self.recording_observer]:
            if rec is None and not rec.has_rgb:
                continue

            result = rec.get_rgb_image(t_ns_global, time_domain=TimeDomain.TIME_CODE)
            if abs(result[-1] / 1e6) > 33:  # 33ms
                logger.warning(f"time difference for image query: {result[-1]} ms")
            data[rec.tag] = result
        return data

    def get_all_pointclouds(self) -> dict[str, np.ndarray]:
        data = {}
        for rec in self.get_existing_recordings():
            if not rec.has_pointcloud:
                continue

            if self.view_cached_points:
                data[rec.tag] = rec.get_pointcloud_cached(
                    th_dep=self.th_dep,
                    th_invdep=self.th_invdep,
                    max_point_count=self.max_point_count,
                )
            else:
                data[rec.tag] = rec.get_pointcloud(
                    th_dep=self.th_dep,
                    th_invdep=self.th_invdep,
                    max_point_count=self.max_point_count,
                )
        return data

    def get_all_trajectories(self) -> dict[str, np.ndarray]:
        data = {}
        for rec in self.get_existing_recordings():
            if rec.has_vrs and rec.has_pose:
                data[rec.tag] = rec.sample_trajectory_world_device(
                    sample_fps=self.trajectory_sample_fps
                )
        return data

    def get_synced_poses(self, t_ns_global: int) -> dict[str, any]:
        data = {}
        T_Wd_Hd = None
        for rec in self.get_existing_recordings():
            if rec is None or not rec.has_pose:
                continue

            pose: ClosedLoopTrajectoryPose = None
            tdiff: int = None
            pose, tdiff = rec.get_pose(t_ns_global, time_domain=TimeDomain.TIME_CODE)
            if abs(tdiff / 1e6) > 2:  # 2ms
                logger.warning(f"time difference for pose query {tdiff / 1e6} ms")

            data[rec.tag] = pose
            if rec.tag == "recording_head":
                T_Wd_Hd: SE3 = pose.transform_world_device

        if self.body_dp is not None:
            if self.body_model == BodyModel.MOMENTUM:
                # Momentum path: apply hand-eye alignment and return skeleton + skin
                if self.recording_head is not None and T_Wd_Hd is not None:
                    T_Wd_Hx = T_Wd_Hd @ self.T_Hd_Hx(t_ns_global)
                    t_us = t_ns_global / 1e3
                    skel, skin = self.body_dp.get_posed_skeleton_and_skin(
                        t_us, T_W_Hx=T_Wd_Hx
                    )
                    data["xsens"] = skel
                    if skin is not None:
                        data["momentum"] = skin
            elif self.body_model == BodyModel.SMPL:
                # SMPL vertices are already in Aria world space
                frame_idx = self._get_frame_idx(t_ns_global)
                data["smpl"] = self.body_dp.get_posed_skin(frame_idx)
            elif self.body_model == BodyModel.MHR:
                # MHR vertices are already in Aria world space
                frame_idx = self._get_frame_idx(t_ns_global)
                data["mhr"] = self.body_dp.get_posed_skin(frame_idx)
        return data

    def _get_frame_idx(self, t_ns_global: int) -> int:
        """Convert a global timestamp (nanoseconds) to a body frame index.

        For SMPL, uses the body provider's own timespan.
        For MHR (no stored timestamps), uses the Aria recording timespan.
        """
        total_frames = self.body_dp.num_frames

        if isinstance(self.body_dp, SmplDataProvider):
            t_start_us, t_end_us = self.body_dp.get_global_timespan_us()
            t_start = t_start_us * 1e3  # convert to ns
            t_end = t_end_us * 1e3
        else:
            # MHR: use the recording timespan
            t_start, t_end = float(self.timespan_ns[0]), float(self.timespan_ns[1])

        t_ns = float(t_ns_global)
        if t_end <= t_start:
            return 0
        if t_ns <= t_start:
            return 0
        if t_ns >= t_end:
            return total_frames - 1
        frac = (t_ns - t_start) / (t_end - t_start)
        return min(int(frac * total_frames), total_frames - 1)

    def __compute_xsens_to_aria_alignment(self) -> None:
        """
        \brief Compute se3 transform from xsens head to aria head
               This function will set self.Ts_Hd_Hx and self.t_ns_align
               Only runs for MOMENTUM body model.  SMPL and MHR are already
               in Aria world coordinates and need no alignment.
        """
        if (
            self.recording_head is None
            or self.body_dp is None
            or self.body_model != BodyModel.MOMENTUM
        ):
            self.Ts_Hd_Hx = [SE3.from_matrix(np.eye(4))]
            self.t_ns_align = None
            return
        else:
            logger.info("compute alignment from xsens head to aria headset")
            assert self.body_dp is not None
            assert self.recording_head is not None

        # get synchronized trajectory
        xsens_traj = self.body_dp.get_T_w_h(self.timespan_ns)
        T_Wx_Hx: list[SE3] = xsens_traj[0]
        t_ns: list[int] = xsens_traj[-1]
        T_Wd_Hd: list[SE3] = []
        for t in t_ns:
            pose, _ = self.recording_head.get_pose(t, TimeDomain.TIME_CODE)
            T_Wd_Hd.append(pose.transform_world_device)

        # solve handeye
        handeye = HandEyeSolver(
            stride=self.handeye_stride,
            smooth=self.handeye_smooth,
            skip=self.handeye_skip,
            window=self.handeye_window,
        )
        self.Ts_Hd_Hx: list[SE3] = handeye(
            T_Wa_A=T_Wd_Hd,
            T_Wb_B=T_Wx_Hx,
        )
        if len(self.Ts_Hd_Hx) > 1:
            self.t_ns_align = t_ns[0 :: self.handeye_skip]
        else:
            self.t_ns_align = None

    def T_Hd_Hx(self, t_ns: int) -> SE3:
        if self.t_ns_align is None:
            return self.Ts_Hd_Hx[0]

        if t_ns <= self.t_ns_align[0]:
            return self.Ts_Hd_Hx[0]

        if t_ns >= self.t_ns_align[-1]:
            return self.Ts_Hd_Hx[-1]

        idx = np.searchsorted(self.t_ns_align, t_ns)
        return self.Ts_Hd_Hx[idx]
