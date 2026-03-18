# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from nymeria.definitions import Subpaths, VrsFiles, SlamFiles
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps import (
    ClosedLoopTrajectoryPose,
    MpsDataPathsProvider,
    MpsDataProvider,
    OnlineCalibration,
)
from projectaria_tools.core.sensor_data import (
    ImageData,
    ImageDataRecord,
    TimeDomain,
    TimeQueryOptions,
)
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId


class AriaStream(Enum):
    camera_slam_left = "1201-1"
    camera_slam_right = "1201-2"
    camera_rgb = "214-1"
    imu_right = "1202-1"
    imu_left = "1202-2"


# Type alias for online calibration lookup strategy
OnlineCalibLookupStrategy = Literal["nearest", "interpolate"]

# Sensor labels that have online calibration in MPS output
# (IMUs and cameras only; mag/baro/microphones do not)
LABELS_HAS_ONLINE_CALIB_SENSOR: set[str] = {
    "imu-left",
    "imu-right",
    "camera-slam-left",
    "camera-slam-right",
    "camera-rgb",
}

# Sensor labels that have factory-calibrated extrinsics.
# Sensors not in this set require use_cad_calib=True for extrinsics.
LABELS_HAS_FACTORY_EXTRINSICS: set[str] = {
    "imu-left",
    "imu-right",
    "camera-slam-left",
    "camera-slam-right",
    "camera-rgb",
    "camera-et-left",
    "camera-et-right",
}


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

    @property
    def online_calibration_path(self) -> Path | None:
        """Path to online_calibration.jsonl file"""
        path = self.recording_path / SlamFiles.online_calibration
        if path.is_file():
            return path
        return None


class RecordingDataProvider(RecordingPathProvider):
    def __init__(
        self,
        recording_path: Path,
        load_online_calib: bool = False,
        online_calib_lookup_strategy: OnlineCalibLookupStrategy = "nearest",
    ) -> None:
        super().__init__(recording_path)

        self._load_online_calib = load_online_calib
        self._online_calib_lookup_strategy = online_calib_lookup_strategy

        self._vrs_dp = None
        self._mps_dp = None
        self._online_calibs: list[OnlineCalibration] | None = None
        self._online_calib_timestamps_ns: np.ndarray | None = None

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

        # load online calibration if requested
        if load_online_calib and self.online_calibration_path is not None:
            self._load_online_calibration()

    def _load_online_calibration(self) -> None:
        """Load online calibration from jsonl file"""
        if self.online_calibration_path is None:
            logger.warning(
                f"Online calibration file not found at {self.recording_path}"
            )
            return

        self._online_calibs = mps.read_online_calibration(
            str(self.online_calibration_path)
        )

        if len(self._online_calibs) == 0:
            logger.warning("Online calibration file is empty")
            self._online_calibs = None
            return

        # Pre-compute timestamps in nanoseconds for fast lookup
        # tracking_timestamp is a timedelta, convert to ns
        self._online_calib_timestamps_ns = np.array(
            [
                int(calib.tracking_timestamp.total_seconds() * 1e9)
                for calib in self._online_calibs
            ]
        )
        logger.info(
            f"Loaded {len(self._online_calibs)} online calibrations "
            f"from {self.online_calibration_path}"
        )

    @property
    def vrs_dp(self) -> VrsDataProvider | None:
        return self._vrs_dp

    @property
    def mps_dp(self) -> MpsDataProvider | None:
        return self._mps_dp

    @property
    def has_online_calibration(self) -> bool:
        """Check if online calibration data is loaded and available"""
        return self._online_calibs is not None and len(self._online_calibs) > 0

    @property
    def online_calib_lookup_strategy(self) -> OnlineCalibLookupStrategy:
        """Return the current online calibration lookup strategy"""
        return self._online_calib_lookup_strategy

    def get_global_timespan_ns(self) -> tuple[int, int]:
        if self.vrs_dp is None:
            raise RuntimeError(
                f"require {self.data_vrsfile=} or {self.motion_vrsfile=}"
            )

        t_start = self.vrs_dp.get_first_time_ns_all_streams(TimeDomain.TIME_CODE)
        t_end = self.vrs_dp.get_last_time_ns_all_streams(TimeDomain.TIME_CODE)
        return t_start, t_end

    # ---- Online calibration lookup ----

    def get_online_calibration_at_timestamp(
        self, t_ns: int, time_domain: TimeDomain = TimeDomain.TIME_CODE
    ) -> OnlineCalibration:
        """
        Get online calibration closest to given timestamp.

        Args:
            t_ns: Timestamp in nanoseconds
            time_domain: Time domain of t_ns (DEVICE_TIME or TIME_CODE)

        Returns:
            OnlineCalibration object at the closest timestamp

        Raises:
            RuntimeError: If online calibration is not loaded
        """
        if self._online_calibs is None or self._online_calib_timestamps_ns is None:
            raise RuntimeError("Online calibration not loaded")

        # Convert to device time if needed (online calib timestamps are in device time)
        if time_domain == TimeDomain.TIME_CODE and self._vrs_dp is not None:
            t_ns_device = self._vrs_dp.convert_from_timecode_to_device_time_ns(t_ns)
        else:
            t_ns_device = t_ns

        # Find nearest calibration using binary search
        idx = np.searchsorted(self._online_calib_timestamps_ns, t_ns_device)

        # Clamp to valid range and find closest
        if idx == 0:
            return self._online_calibs[0]
        if idx >= len(self._online_calibs):
            return self._online_calibs[-1]

        # Check which neighbor is closer
        t_left = self._online_calib_timestamps_ns[idx - 1]
        t_right = self._online_calib_timestamps_ns[idx]
        if abs(t_left - t_ns_device) <= abs(t_right - t_ns_device):
            return self._online_calibs[idx - 1]
        return self._online_calibs[idx]

    # ---- Sensor calibration access ----

    def get_sensor_calibration(
        self,
        sensor_label: str,
        t_ns: int | None = None,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
    ):
        """
        Get sensor calibration (intrinsics + extrinsics) for a given sensor label.

        If load_online_calib=True and t_ns is provided and the sensor has online
        calibration, returns the time-varying calibration at the closest timestamp.
        Otherwise returns factory/CAD calibration.

        For sensors that should have online calibration (IMUs, cameras) but don't
        because online calib was requested but not loaded: raises RuntimeError
        (no silent fallback).

        Args:
            sensor_label: Sensor label (e.g. "imu-left", "camera-rgb", "mag0")
            t_ns: Optional timestamp for online calibration lookup (nanoseconds)
            time_domain: Time domain of t_ns

        Returns:
            Sensor-specific calibration object:
                - ImuCalibration for "imu-left", "imu-right"
                - CameraCalibration for camera sensors
                - MagnetometerCalibration for "mag0"
                - etc.

        Raises:
            RuntimeError: If online calib was requested but is not available for
                a sensor that should have it
            ValueError: If the sensor label is not found in calibration data
        """
        has_online = sensor_label in LABELS_HAS_ONLINE_CALIB_SENSOR

        # Online calibration path
        if self._load_online_calib and has_online and t_ns is not None:
            if not self.has_online_calibration:
                raise RuntimeError(
                    f"Online calibration was requested but is not available. "
                    f"Cannot provide online calibration for sensor '{sensor_label}'"
                )

            online_calib = self.get_online_calibration_at_timestamp(t_ns, time_domain)

            # Search in IMU calibrations
            if sensor_label.startswith("imu-"):
                for imu_calib in online_calib.imu_calibs:
                    if imu_calib.get_label() == sensor_label:
                        return imu_calib
                raise ValueError(
                    f"Sensor '{sensor_label}' not found in online IMU calibrations"
                )

            # Search in camera calibrations
            if sensor_label.startswith("camera-"):
                for cam_calib in online_calib.camera_calibs:
                    if cam_calib.get_label() == sensor_label:
                        return cam_calib
                raise ValueError(
                    f"Sensor '{sensor_label}' not found in online camera calibrations"
                )

        # Factory / CAD calibration fallback
        if self._vrs_dp is None:
            raise RuntimeError(
                "VRS data provider required for factory/CAD calibration"
            )
        device_calib = self._vrs_dp.get_device_calibration()
        if device_calib is None:
            raise RuntimeError("Device calibration not available in VRS file")

        sensor_calib = device_calib.get_sensor_calib(sensor_label)
        if sensor_calib is None:
            raise ValueError(
                f"Sensor '{sensor_label}' not found in device calibration"
            )
        return sensor_calib

    def get_T_device_sensor(
        self,
        sensor_label: str,
        t_ns: int | None = None,
        time_domain: TimeDomain = TimeDomain.TIME_CODE,
    ) -> SE3:
        """
        Get SE3 transformation from device frame to sensor frame.

        Uses online calibration (time-varying) if available and requested,
        otherwise falls back to factory calibration, or CAD calibration
        for sensors without factory extrinsics (e.g. magnetometer, barometer).

        Args:
            sensor_label: Sensor label (e.g. "imu-left", "camera-rgb", "mag0")
            t_ns: Optional timestamp for online calibration lookup (nanoseconds)
            time_domain: Time domain of t_ns

        Returns:
            SE3 transformation T_device_sensor
        """
        has_online = sensor_label in LABELS_HAS_ONLINE_CALIB_SENSOR

        # Online calibration path for sensors that have it
        if self._load_online_calib and has_online and t_ns is not None:
            if not self.has_online_calibration:
                raise RuntimeError(
                    f"Online calibration was requested but is not available. "
                    f"Cannot provide online extrinsics for sensor '{sensor_label}'"
                )

            online_calib = self.get_online_calibration_at_timestamp(t_ns, time_domain)

            if sensor_label.startswith("imu-"):
                for imu_calib in online_calib.imu_calibs:
                    if imu_calib.get_label() == sensor_label:
                        return imu_calib.get_transform_device_imu()
                raise ValueError(
                    f"Sensor '{sensor_label}' not found in online IMU calibrations"
                )

            if sensor_label.startswith("camera-"):
                for cam_calib in online_calib.camera_calibs:
                    if cam_calib.get_label() == sensor_label:
                        return cam_calib.get_transform_device_camera()
                raise ValueError(
                    f"Sensor '{sensor_label}' not found in online camera calibrations"
                )

        # Factory / CAD calibration fallback
        if self.vrs_dp is None:
            raise RuntimeError(
                "VRS data provider required for factory/CAD extrinsics"
            )
        device_calib = self.vrs_dp.get_device_calibration()
        if device_calib is None:
            raise RuntimeError("Device calibration not available in VRS file")

        use_cad = sensor_label not in LABELS_HAS_FACTORY_EXTRINSICS
        T_device_sensor = device_calib.get_transform_device_sensor(
            sensor_label, use_cad
        )
        if T_device_sensor is None:
            raise ValueError(
                f"Extrinsics for sensor '{sensor_label}' not found in device "
                f"calibration (use_cad_calib={use_cad})"
            )
        return T_device_sensor

    # ---- Sensor timestamps ----

    def get_sensor_data_with_device_time_ns(self, sensor_label: str) -> np.ndarray:
        """
        Get all native capture timestamps for a sensor stream from VRS.

        Iterates over the raw sensor data in the VRS file and extracts
        the capture timestamp for each sample, at the sensor's native rate.

        Timestamps are returned in DEVICE_TIME domain. Use
        vrs_dp.convert_from_device_time_to_timecode_ns() to convert
        individual timestamps to TIME_CODE if needed for cross-device
        comparisons.

        Args:
            sensor_label: Sensor label (e.g. "imu-left", "imu-right")

        Returns:
            np.ndarray of int64 timestamps in nanoseconds (DEVICE_TIME).

        Raises:
            RuntimeError: If VRS data provider is not available
        """
        if self._vrs_dp is None:
            raise RuntimeError("VRS data provider required for sensor timestamps")

        stream_id = self._vrs_dp.get_stream_id_from_label(sensor_label)
        n = self._vrs_dp.get_num_data(stream_id)

        # timestamps = np.empty(n, dtype=np.int64)
        all_imu_data = [ None ] * n
        for i in range(n):
            imu_data = self._vrs_dp.get_imu_data_by_index(stream_id, i)
            # timestamps[i] = imu_data.capture_timestamp_ns
            all_imu_data[i] = imu_data

        # return timestamps, imu_data
        return all_imu_data

    # ---- Point cloud ----

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

    # ---- VRS / RGB ----

    @property
    def has_vrs(self) -> bool:
        return self._vrs_dp is not None

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

    # ---- Pose (T_world_device) ----

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
    load_online_calib: bool = False,
    online_calib_lookup_strategy: OnlineCalibLookupStrategy = "nearest",
) -> RecordingDataProvider | None:
    """
    Factory function to create a RecordingDataProvider.

    Args:
        recording_path: Path to the recording directory
        load_online_calib: Whether to load online calibration data
        online_calib_lookup_strategy: Strategy for looking up online calibration
            by timestamp ("nearest" or "interpolate"). Currently only "nearest"
            is implemented.

    Returns:
        RecordingDataProvider instance or None if the path is invalid
    """
    if not recording_path.is_dir():
        return None

    dp = RecordingDataProvider(
        recording_path,
        load_online_calib=load_online_calib,
        online_calib_lookup_strategy=online_calib_lookup_strategy,
    )
    if dp.vrs_dp is None and dp.mps_dp is None:
        return None
    else:
        return dp
