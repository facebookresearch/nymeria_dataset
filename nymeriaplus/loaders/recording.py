# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for one Aria recording (VRS + MPS)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import (
    ClosedLoopTrajectoryPose,
    MpsDataPathsProvider,
    MpsDataProvider,
)
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)
from projectaria_tools.core.stream_id import StreamId
from tqdm import tqdm

logger = logging.getLogger(__name__)

_RGB_STREAM_ID = StreamId("214-1")
_POINTS_CACHE_NAME = "semidense_points_cache.npz"
_REL_DATA_VRS = "data/data.vrs"
_REL_MOTION_VRS = "data/motion.vrs"
_REL_MPS_DIR = "mps"
_REL_MPS_SLAM_DIR = "mps/slam"


class RecordingLoader:
    """Wraps one Aria recording. Never raises in ``__init__`` -- inspect
    :attr:`is_valid` after construction.
    """

    def __init__(self, recording_path: Path) -> None:
        self.recording_path = recording_path
        self.tag = recording_path.name
        self._vrs: VrsDataProvider | None = None
        self._mps: MpsDataProvider | None = None

        if not recording_path.is_dir():
            logger.warning(f"recording dir missing: {recording_path}")
            return

        self._vrs = self._open_vrs()
        self._mps = self._open_mps()

    # ------------------------------------------------------------------ open

    def _open_vrs(self) -> VrsDataProvider | None:
        for rel in (_REL_DATA_VRS, _REL_MOTION_VRS):
            p = self.recording_path / rel
            if p.is_file():
                return data_provider.create_vrs_data_provider(str(p))
        logger.warning(f"{self.tag}: no VRS file found")
        return None

    def _open_mps(self) -> MpsDataProvider | None:
        mps_dir = self.recording_path / _REL_MPS_DIR
        if not mps_dir.is_dir():
            logger.warning(f"{self.tag}: no MPS dir at {mps_dir}")
            return None
        return MpsDataProvider(MpsDataPathsProvider(str(mps_dir)).get_data_paths())

    # ------------------------------------------------------------------ flags

    @property
    def is_valid(self) -> bool:
        return self._vrs is not None or self._mps is not None

    @property
    def vrs(self) -> VrsDataProvider | None:
        return self._vrs

    @property
    def mps(self) -> MpsDataProvider | None:
        return self._mps

    @property
    def has_vrs(self) -> bool:
        return self._vrs is not None

    @property
    def has_rgb(self) -> bool:
        return self.has_vrs and self._vrs.check_stream_is_active(_RGB_STREAM_ID)

    @property
    def has_pose(self) -> bool:
        return self._mps is not None and self._mps.has_closed_loop_poses()

    @property
    def has_pointcloud(self) -> bool:
        return self._mps is not None and self._mps.has_semidense_point_cloud()

    @property
    def points_cache_path(self) -> Path:
        return self.recording_path / _REL_MPS_SLAM_DIR / _POINTS_CACHE_NAME

    # ------------------------------------------------------------------ time

    def get_global_timespan_ns(self) -> tuple[int, int]:
        if self._vrs is None:
            raise RuntimeError(f"{self.tag} has no VRS")
        t0 = self._vrs.get_first_time_ns_all_streams(TimeDomain.TIME_CODE)
        t1 = self._vrs.get_last_time_ns_all_streams(TimeDomain.TIME_CODE)
        return t0, t1

    def to_device_time_ns(self, t_ns: int, time_domain: TimeDomain) -> int:
        if time_domain == TimeDomain.DEVICE_TIME:
            return t_ns
        if time_domain == TimeDomain.TIME_CODE:
            if self._vrs is None:
                raise RuntimeError("VRS required for TIME_CODE conversion")
            return self._vrs.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )
        raise ValueError(f"unsupported time domain {time_domain}")

    # ------------------------------------------------------------------ rgb

    def get_rgb_image(
        self, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> tuple[ImageData, ImageDataRecord, int]:
        if not self.has_rgb:
            raise RuntimeError(f"{self.tag} has no RGB stream")
        t_dev = self.to_device_time_ns(t_ns, time_domain)
        image_data, image_meta = self._vrs.get_image_data_by_time_ns(
            _RGB_STREAM_ID,
            time_ns=t_dev,
            time_domain=TimeDomain.DEVICE_TIME,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        return image_data, image_meta, t_dev - image_meta.capture_timestamp_ns

    # ------------------------------------------------------------------ pose

    def get_pose(
        self, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> tuple[ClosedLoopTrajectoryPose, int]:
        if not self.has_pose:
            raise RuntimeError(f"{self.tag} has no pose")
        t_dev = self.to_device_time_ns(int(t_ns), time_domain)
        pose = self._mps.get_closed_loop_pose(t_dev, TimeQueryOptions.CLOSEST)
        tdiff = int(pose.tracking_timestamp.total_seconds() * 1e9 - t_dev)
        return pose, tdiff

    def sample_trajectory_at_timecodes(
        self, sample_times_ns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised pose sampling for synchronization. Returns (poses Nx4x4,
        tdiffs N)."""
        n = len(sample_times_ns)
        poses = np.empty((n, 4, 4), dtype=np.float32)
        tdiffs = np.empty(n, dtype=np.int64)
        for i in tqdm(range(n), desc=f"  {self.tag}", leave=False):
            pose, tdiff = self.get_pose(int(sample_times_ns[i]), TimeDomain.TIME_CODE)
            poses[i] = pose.transform_world_device.to_matrix().astype(np.float32)
            tdiffs[i] = tdiff
        return poses, tdiffs

    # ------------------------------------------------------------------ pcl

    def get_pointcloud(
        self,
        th_invdep: float = 0.0004,
        th_dep: float = 0.02,
        max_point_count: int | None = 50_000,
        cache: bool = False,
    ) -> np.ndarray:
        if not self.has_pointcloud:
            raise RuntimeError(f"{self.tag} has no pointcloud")
        points = self._mps.get_semidense_point_cloud()
        points = mps.utils.filter_points_from_confidence(
            raw_points=points, threshold_dep=th_dep, threshold_invdep=th_invdep
        )
        if max_point_count is not None:
            points = mps.utils.filter_points_from_count(
                raw_points=points, max_point_count=max_point_count
            )
        arr = np.array([x.position_world for x in points])
        if cache:
            np.savez(
                self.points_cache_path,
                points=arr,
                threshold_dep=th_dep,
                threshold_invdep=th_invdep,
                max_point_count=max_point_count,
            )
        return arr

    def get_pointcloud_cached(
        self,
        th_invdep: float = 0.0004,
        th_dep: float = 0.02,
        max_point_count: int | None = 50_000,
    ) -> np.ndarray:
        if self.points_cache_path.is_file():
            logger.info(f"load cached pointcloud {self.points_cache_path}")
            return np.load(self.points_cache_path)["points"]
        return self.get_pointcloud(
            th_invdep=th_invdep,
            th_dep=th_dep,
            max_point_count=max_point_count,
            cache=True,
        )
