# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger

from nymeria.definitions import Subpaths, VrsFiles
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


class AriaStream(Enum):
    camera_slam_left = "1201-1"
    camera_slam_right = "1201-2"
    camera_rgb = "214-1"
    imu_right = "1202-1"
    imu_left = "1202-2"


class RecordingPathProvider:
    """
    \brief This class will not check of input recording path is valid
    """

    def __init__(self, recording_path: Path):
        self.recording_path: Path = recording_path
        self.tag: str = recording_path.name

    @property
    def data_vrsfile(self) -> Path:
        return self.recording_path / VrsFiles.data

    @property
    def motion_vrsfile(self) -> Path:
        return self.recording_path / VrsFiles.motion

    @property
    def mps_path(self) -> MpsDataPathsProvider | None:
        mps_path = self.recording_path / Subpaths.mps
        if mps_path.is_dir():
            return MpsDataPathsProvider(str(mps_path))
        else:
            return None

    @property
    def points_npz_cache(self) -> Path:
        return self.recording_path / Subpaths.mps_slam / "semidense_points_cache.npz"


class RecordingDataProvider(RecordingPathProvider):
    def __init__(self, recording_path: Path) -> None:
        super().__init__(recording_path)

        self._vrs_dp = None
        self._mps_dp = None
        if not self.recording_path.is_dir():
            return

        # load vrs
        if self.data_vrsfile.is_file():
            self._vrs_dp = data_provider.create_vrs_data_provider(
                str(self.data_vrsfile)
            )
        elif self.motion_vrsfile.is_file():
            self._vrs_dp = data_provider.create_vrs_data_provider(
                str(self.motion_vrsfile)
            )

        # load mps
        if self.mps_path is not None:
            self._mps_dp = MpsDataProvider(self.mps_path.get_data_paths())

    @property
    def vrs_dp(self) -> VrsDataProvider | None:
        return self._vrs_dp

    @property
    def mps_dp(self) -> MpsDataProvider | None:
        return self._mps_dp

    def get_global_timespan_ns(self) -> tuple[int, int]:
        if self.vrs_dp is None:
            raise RuntimeError(
                f"require {self.data_vrsfile=} or {self.motion_vrsfile=}"
            )

        t_start = self.vrs_dp.get_first_time_ns_all_streams(TimeDomain.TIME_CODE)
        t_end = self.vrs_dp.get_last_time_ns_all_streams(TimeDomain.TIME_CODE)
        return t_start, t_end

    @property
    def has_pointcloud(self) -> bool:
        if self.mps_dp is None or not self.mps_dp.has_semidense_point_cloud():
            return False
        else:
            return True

    def get_pointcloud(
        self,
        th_invdep: float = 0.0004,
        th_dep: float = 0.02,
        max_point_count: int = 50_000,
        cache_to_npz: bool = False,
    ) -> np.ndarray:
        assert self.has_pointcloud, "recording has no point cloud"
        points = self.mps_dp.get_semidense_point_cloud()

        points = mps.utils.filter_points_from_confidence(
            raw_points=points, threshold_dep=th_dep, threshold_invdep=th_invdep
        )
        points = mps.utils.filter_points_from_count(
            raw_points=points, max_point_count=max_point_count
        )

        points = np.array([x.position_world for x in points])

        if cache_to_npz:
            np.savez(
                self.points_npz_cache,
                points=points,
                threshold_dep=th_dep,
                threshold_invdep=th_invdep,
                max_point_count=max_point_count,
            )
        return points

    def get_pointcloud_cached(
        self,
        th_invdep: float = 0.0004,
        th_dep: float = 0.02,
        max_point_count: int = 50_000,
    ) -> np.ndarray:
        assert self.has_pointcloud, "recording has no point cloud"
        if self.points_npz_cache.is_file():
            logger.info(f"load cached point cloud from {self.points_npz_cache}")
            return np.load(self.points_npz_cache)["points"]

        return self.get_pointcloud(cache_to_npz=True)

    @property
    def has_vrs(self) -> bool:
        return self.vrs_dp is not None

    @property
    def has_rgb(self) -> bool:
        return self.has_vrs and self.vrs_dp.check_stream_is_active(StreamId("214-1"))

    def get_rgb_image(
        self, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> tuple[ImageData, ImageDataRecord, int]:
        assert self.has_rgb, "recording has no rgb video"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )
        else:
            t_ns_device = t_ns

        image_data, image_meta = self.vrs_dp.get_image_data_by_time_ns(
            StreamId("214-1"),
            time_ns=t_ns_device,
            time_domain=TimeDomain.DEVICE_TIME,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        t_diff = t_ns_device - image_meta.capture_timestamp_ns

        return image_data, image_meta, t_diff

    @property
    def has_pose(self) -> bool:
        if self.mps_dp is None or not self.mps_dp.has_closed_loop_poses():
            return False
        else:
            return True

    def get_pose(
        self, t_ns: int, time_domain: TimeDomain
    ) -> tuple[ClosedLoopTrajectoryPose, int]:
        t_ns = int(t_ns)
        assert self.has_pose, "recording has no closed loop trajectory"
        assert time_domain in [
            TimeDomain.DEVICE_TIME,
            TimeDomain.TIME_CODE,
        ], "unsupported time domain"

        if time_domain == TimeDomain.TIME_CODE:
            assert self.vrs_dp, "require vrs for time domain mapping"
            t_ns_device = self.vrs_dp.convert_from_timecode_to_device_time_ns(
                timecode_time_ns=t_ns
            )

        else:
            t_ns_device = t_ns

        pose = self.mps_dp.get_closed_loop_pose(t_ns_device, TimeQueryOptions.CLOSEST)
        t_diff = pose.tracking_timestamp.total_seconds() * 1e9 - t_ns_device
        return pose, t_diff

    def sample_trajectory_world_device(self, sample_fps: float = 1) -> np.ndarray:
        assert self.has_pose, "recording has no closed loop trajectory"
        assert self.has_vrs, "current implementation assume vrs is loaded."
        t_start, t_end = self.get_global_timespan_ns()
        t_start = self.vrs_dp.convert_from_timecode_to_device_time_ns(t_start)
        t_end = self.vrs_dp.convert_from_timecode_to_device_time_ns(t_end)

        dt = int(1e9 / sample_fps)
        traj_world_device = []
        for t_ns in range(t_start, t_end, dt):
            pose = self.mps_dp.get_closed_loop_pose(t_ns, TimeQueryOptions.CLOSEST)
            traj_world_device.append(
                pose.transform_world_device.to_matrix().astype(np.float32)
            )

        traj_world_device = np.stack(traj_world_device, axis=0)
        return traj_world_device


def create_recording_data_provider(
    recording_path: Path,
) -> RecordingDataProvider | None:
    if not recording_path.is_dir():
        return None

    dp = RecordingDataProvider(recording_path)
    if dp.vrs_dp is None and dp.mps_dp is None:
        return None
    else:
        return dp
