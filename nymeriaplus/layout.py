# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sequence on-disk layout, body model enum, and data-group definitions.

Sequence root structure::

    folder/
    ├── LICENSE
    ├── metadata.json
    ├── body/
    │   ├── xdata.healthcheck
    │   ├── xdata.mvnx
    │   ├── xdata.npz
    │   ├── xdata_mhr.glb
    │   └── xdata_smpl_neutral.npz
    ├── narration/
    │   ├── activity_summarization.csv
    │   ├── atomic_action.csv
    │   └── motion_narration.csv
    ├── objects/
    │   ├── boxy/
    │   │   ├── 2dbb_recording_head.csv
    │   │   ├── 2dbb_recording_lwrist.csv
    │   │   ├── 2dbb_recording_observer.csv
    │   │   ├── 2dbb_recording_rwrist.csv
    │   │   ├── 3dbb.csv
    │   │   ├── instances.json
    │   │   └── scene_objects.csv
    │   └── shaper/
    │       ├── list of *.ply files
    │       └── shaper_metadata.csv
    ├── recording_head/
    │   ├── data/
    │   │   ├── audio.vrs
    │   │   ├── data.vrs
    │   │   ├── et.vrs
    │   │   └── motion.vrs
    │   └── mps/
    │       ├── eye_gaze/
    │       │   ├── general_eye_gaze.csv
    │       │   └── personalized_eye_gaze.csv
    │       └── slam/
    │           ├── closed_loop_trajectory.csv
    │           ├── online_calibration.jsonl
    │           ├── open_loop_trajectory.csv
    │           ├── semidense_observations.csv.gz
    │           ├── semidense_points.csv.gz
    │           └── summary.json
    ├── recording_lwrist/  (and recording_rwrist/, no eye_gaze, no audio/et VRS)
    │   ├── data/{data.vrs, motion.vrs}
    │   └── mps/slam/{...}
    └── recording_observer/  (same as recording_head/)


Data groups (downloadable bundles)
----------------------------------

Each download URL maps to one data group -- one URL per (sequence, group).
Multi-file groups are packed into zips by the storage layer; entries that
point at a directory expand to every file under it.

================================  ========================================================
Group                             Contents
================================  ========================================================
``body_raw``                      ``body/xdata.{healthcheck,mvnx,npz}``
``body_processed``                ``body/xdata_mhr.glb`` + ``body/xdata_smpl_neutral.npz``
``slam``                          Per-recording closed/open-loop trajectories,
                                  semidense points, online calibration, summary.
``slam_semidense_observations``   Per-recording ``semidense_observations.csv.gz`` (large).
``timesync_and_imu``              Per-recording ``data/motion.vrs``.
``audio``                         Headset audio for head + observer.
``eye_tracking``                  ET VRS + ``mps/eye_gaze/`` for head + observer.
``object_bounding_box``           ``objects/boxy/`` -- 2D/3D bbox + scene objects.
``object_mesh``                   ``objects/shaper/`` -- per-instance ShapeR ``.ply``.
``narration``                     ``narration/`` -- motion narration + actions + summary.
================================  ========================================================

Files in :data:`UNGROUPED_FILES` (LICENSE, metadata.json, the four primary
``data.vrs`` files) are individually selectable but **not** auto-included.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar


# ---------------------------------------------------------------------------
# Body model + per-recording enums
# ---------------------------------------------------------------------------


class BodyModel(Enum):
    SMPL = "smpl"
    MHR = "mhr"


class Recording(Enum):
    """Per-device Aria recording subdirectory names."""

    HEAD = "recording_head"
    OBSERVER = "recording_observer"
    LEFT_WRIST = "recording_lwrist"
    RIGHT_WRIST = "recording_rwrist"


# ---------------------------------------------------------------------------
# Sequence layout: every relative path declared once, in one place.
# ---------------------------------------------------------------------------


class SequenceLayout:
    """Flat namespace of every sequence-relative path in a NymeriaPlus release.

    Use class attributes directly::

        smpl_path = Path(seq_root) / SequenceLayout.BODY_XDATA_SMPL_NPZ

    Or iterate everything::

        for rel in SequenceLayout.all_paths():
            ...
    """

    # fmt: off

    # ----- Top level -----
    LICENSE: ClassVar[str] = "LICENSE"
    METADATA_JSON: ClassVar[str] = "metadata.json"

    # ----- Body -----
    BODY_DIR: ClassVar[str] = "body"
    BODY_XDATA_HEALTHCHECK: ClassVar[str] = "body/xdata.healthcheck"
    BODY_XDATA_MVNX: ClassVar[str] = "body/xdata.mvnx"
    BODY_XDATA_NPZ: ClassVar[str] = "body/xdata.npz"
    BODY_XDATA_MHR_GLB: ClassVar[str] = "body/xdata_mhr.glb"
    BODY_XDATA_SMPL_NPZ: ClassVar[str] = "body/xdata_smpl_neutral.npz"

    # ----- Narration -----
    NARRATION_DIR: ClassVar[str] = "narration"
    NARRATION_ACTIVITY_SUMMARIZATION_CSV: ClassVar[str] = "narration/activity_summarization.csv"
    NARRATION_ATOMIC_ACTION_CSV: ClassVar[str] = "narration/atomic_action.csv"
    NARRATION_MOTION_NARRATION_CSV: ClassVar[str] = "narration/motion_narration.csv"

    # ----- Objects: Boxy (2D/3D bounding boxes) -----
    OBJECTS_BOXY_DIR: ClassVar[str] = "objects/boxy"
    OBJECTS_BOXY_2DBB_HEAD_CSV: ClassVar[str] = "objects/boxy/2dbb_recording_head.csv"
    OBJECTS_BOXY_2DBB_LWRIST_CSV: ClassVar[str] = "objects/boxy/2dbb_recording_lwrist.csv"
    OBJECTS_BOXY_2DBB_OBSERVER_CSV: ClassVar[str] = "objects/boxy/2dbb_recording_observer.csv"
    OBJECTS_BOXY_2DBB_RWRIST_CSV: ClassVar[str] = "objects/boxy/2dbb_recording_rwrist.csv"
    OBJECTS_BOXY_3DBB_CSV: ClassVar[str] = "objects/boxy/3dbb.csv"
    OBJECTS_BOXY_INSTANCES_JSON: ClassVar[str] = "objects/boxy/instances.json"
    OBJECTS_BOXY_SCENE_OBJECTS_CSV: ClassVar[str] = "objects/boxy/scene_objects.csv"

    # ----- Objects: ShapeR (instance-level 3D reconstruction) -----
    OBJECTS_SHAPER_DIR: ClassVar[str] = "objects/shaper"
    OBJECTS_SHAPER_METADATA_CSV: ClassVar[str] = "objects/shaper/shaper_metadata.csv"
    # Note: per-instance .ply files live under OBJECTS_SHAPER_DIR; enumerate at runtime.

    # ----- Recording: Head -----
    HEAD_DATA_VRS: ClassVar[str] = "recording_head/data/data.vrs"
    HEAD_ET_VRS: ClassVar[str] = "recording_head/data/et.vrs"
    HEAD_MOTION_VRS: ClassVar[str] = "recording_head/data/motion.vrs"
    HEAD_AUDIO_VRS: ClassVar[str] = "recording_head/data/audio.vrs"
    HEAD_EYE_GAZE_DIR: ClassVar[str] = "recording_head/mps/eye_gaze"
    HEAD_CLOSED_LOOP_TRAJECTORY: ClassVar[str] = "recording_head/mps/slam/closed_loop_trajectory.csv"
    HEAD_ONLINE_CALIBRATION: ClassVar[str] = "recording_head/mps/slam/online_calibration.jsonl"
    HEAD_OPEN_LOOP_TRAJECTORY: ClassVar[str] = "recording_head/mps/slam/open_loop_trajectory.csv"
    HEAD_SEMIDENSE_OBSERVATIONS: ClassVar[str] = "recording_head/mps/slam/semidense_observations.csv.gz"
    HEAD_SEMIDENSE_POINTS: ClassVar[str] = "recording_head/mps/slam/semidense_points.csv.gz"
    HEAD_SLAM_SUMMARY: ClassVar[str] = "recording_head/mps/slam/summary.json"
    HEAD_GENERAL_EYE_GAZE: ClassVar[str] = "recording_head/mps/eye_gaze/general_eye_gaze.csv"
    HEAD_PERSONALIZED_EYE_GAZE: ClassVar[str] = "recording_head/mps/eye_gaze/personalized_eye_gaze.csv"

    # ----- Recording: Left Wrist -----
    LWRIST_DATA_VRS: ClassVar[str] = "recording_lwrist/data/data.vrs"
    LWRIST_MOTION_VRS: ClassVar[str] = "recording_lwrist/data/motion.vrs"
    LWRIST_CLOSED_LOOP_TRAJECTORY: ClassVar[str] = "recording_lwrist/mps/slam/closed_loop_trajectory.csv"
    LWRIST_ONLINE_CALIBRATION: ClassVar[str] = "recording_lwrist/mps/slam/online_calibration.jsonl"
    LWRIST_OPEN_LOOP_TRAJECTORY: ClassVar[str] = "recording_lwrist/mps/slam/open_loop_trajectory.csv"
    LWRIST_SEMIDENSE_OBSERVATIONS: ClassVar[str] = "recording_lwrist/mps/slam/semidense_observations.csv.gz"
    LWRIST_SEMIDENSE_POINTS: ClassVar[str] = "recording_lwrist/mps/slam/semidense_points.csv.gz"
    LWRIST_SLAM_SUMMARY: ClassVar[str] = "recording_lwrist/mps/slam/summary.json"

    # ----- Recording: Right Wrist -----
    RWRIST_DATA_VRS: ClassVar[str] = "recording_rwrist/data/data.vrs"
    RWRIST_MOTION_VRS: ClassVar[str] = "recording_rwrist/data/motion.vrs"
    RWRIST_CLOSED_LOOP_TRAJECTORY: ClassVar[str] = "recording_rwrist/mps/slam/closed_loop_trajectory.csv"
    RWRIST_ONLINE_CALIBRATION: ClassVar[str] = "recording_rwrist/mps/slam/online_calibration.jsonl"
    RWRIST_OPEN_LOOP_TRAJECTORY: ClassVar[str] = "recording_rwrist/mps/slam/open_loop_trajectory.csv"
    RWRIST_SEMIDENSE_OBSERVATIONS: ClassVar[str] = "recording_rwrist/mps/slam/semidense_observations.csv.gz"
    RWRIST_SEMIDENSE_POINTS: ClassVar[str] = "recording_rwrist/mps/slam/semidense_points.csv.gz"
    RWRIST_SLAM_SUMMARY: ClassVar[str] = "recording_rwrist/mps/slam/summary.json"

    # ----- Recording: Observer -----
    OBSERVER_DATA_VRS: ClassVar[str] = "recording_observer/data/data.vrs"
    OBSERVER_ET_VRS: ClassVar[str] = "recording_observer/data/et.vrs"
    OBSERVER_MOTION_VRS: ClassVar[str] = "recording_observer/data/motion.vrs"
    OBSERVER_AUDIO_VRS: ClassVar[str] = "recording_observer/data/audio.vrs"
    OBSERVER_EYE_GAZE_DIR: ClassVar[str] = "recording_observer/mps/eye_gaze"
    OBSERVER_CLOSED_LOOP_TRAJECTORY: ClassVar[str] = "recording_observer/mps/slam/closed_loop_trajectory.csv"
    OBSERVER_ONLINE_CALIBRATION: ClassVar[str] = "recording_observer/mps/slam/online_calibration.jsonl"
    OBSERVER_OPEN_LOOP_TRAJECTORY: ClassVar[str] = "recording_observer/mps/slam/open_loop_trajectory.csv"
    OBSERVER_SEMIDENSE_OBSERVATIONS: ClassVar[str] = "recording_observer/mps/slam/semidense_observations.csv.gz"
    OBSERVER_SEMIDENSE_POINTS: ClassVar[str] = "recording_observer/mps/slam/semidense_points.csv.gz"
    OBSERVER_SLAM_SUMMARY: ClassVar[str] = "recording_observer/mps/slam/summary.json"
    OBSERVER_GENERAL_EYE_GAZE: ClassVar[str] = "recording_observer/mps/eye_gaze/general_eye_gaze.csv"
    OBSERVER_PERSONALIZED_EYE_GAZE: ClassVar[str] = "recording_observer/mps/eye_gaze/personalized_eye_gaze.csv"
    # fmt: on

    @classmethod
    def all_paths(cls) -> list[str]:
        """Return every sequence-relative path declared on this class."""
        return [
            v
            for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        ]


# ---------------------------------------------------------------------------
# Data groups for download
# ---------------------------------------------------------------------------


class DataGroups(Enum):
    """One atomic downloadable element. One URL per (sequence, group)."""

    BODY_RAW = "body_raw"
    BODY_PROCESSED = "body_processed"
    SLAM = "slam"
    SLAM_SEMIDENSE_OBSERVATIONS = "slam_semidense_observations"
    TIMESYNC_AND_IMU = "timesync_and_imu"
    AUDIO = "audio"
    EYE_TRACKING = "eye_tracking"
    OBJECT_BOUNDING_BOX = "object_bounding_box"
    OBJECT_MESH = "object_mesh"
    NARRATION = "narration"


# Files individually selectable for download but NOT auto-included.
UNGROUPED_FILES: list[str] = [
    SequenceLayout.LICENSE,
    SequenceLayout.METADATA_JSON,
    SequenceLayout.HEAD_DATA_VRS,
    SequenceLayout.LWRIST_DATA_VRS,
    SequenceLayout.RWRIST_DATA_VRS,
    SequenceLayout.OBSERVER_DATA_VRS,
]

_GROUP_FILES: dict[DataGroups, list[str]] = {
    DataGroups.BODY_RAW: [
        SequenceLayout.BODY_XDATA_HEALTHCHECK,
        SequenceLayout.BODY_XDATA_MVNX,
        SequenceLayout.BODY_XDATA_NPZ,
    ],
    DataGroups.BODY_PROCESSED: [
        SequenceLayout.BODY_XDATA_MHR_GLB,
        SequenceLayout.BODY_XDATA_SMPL_NPZ,
    ],
    DataGroups.SLAM: [
        SequenceLayout.HEAD_CLOSED_LOOP_TRAJECTORY,
        SequenceLayout.HEAD_OPEN_LOOP_TRAJECTORY,
        SequenceLayout.HEAD_SEMIDENSE_POINTS,
        SequenceLayout.HEAD_SLAM_SUMMARY,
        SequenceLayout.HEAD_ONLINE_CALIBRATION,
        SequenceLayout.LWRIST_CLOSED_LOOP_TRAJECTORY,
        SequenceLayout.LWRIST_OPEN_LOOP_TRAJECTORY,
        SequenceLayout.LWRIST_SEMIDENSE_POINTS,
        SequenceLayout.LWRIST_SLAM_SUMMARY,
        SequenceLayout.LWRIST_ONLINE_CALIBRATION,
        SequenceLayout.RWRIST_CLOSED_LOOP_TRAJECTORY,
        SequenceLayout.RWRIST_OPEN_LOOP_TRAJECTORY,
        SequenceLayout.RWRIST_SEMIDENSE_POINTS,
        SequenceLayout.RWRIST_SLAM_SUMMARY,
        SequenceLayout.RWRIST_ONLINE_CALIBRATION,
        SequenceLayout.OBSERVER_CLOSED_LOOP_TRAJECTORY,
        SequenceLayout.OBSERVER_OPEN_LOOP_TRAJECTORY,
        SequenceLayout.OBSERVER_SEMIDENSE_POINTS,
        SequenceLayout.OBSERVER_SLAM_SUMMARY,
        SequenceLayout.OBSERVER_ONLINE_CALIBRATION,
    ],
    DataGroups.SLAM_SEMIDENSE_OBSERVATIONS: [
        SequenceLayout.HEAD_SEMIDENSE_OBSERVATIONS,
        SequenceLayout.LWRIST_SEMIDENSE_OBSERVATIONS,
        SequenceLayout.RWRIST_SEMIDENSE_OBSERVATIONS,
        SequenceLayout.OBSERVER_SEMIDENSE_OBSERVATIONS,
    ],
    DataGroups.TIMESYNC_AND_IMU: [
        SequenceLayout.HEAD_MOTION_VRS,
        SequenceLayout.LWRIST_MOTION_VRS,
        SequenceLayout.RWRIST_MOTION_VRS,
        SequenceLayout.OBSERVER_MOTION_VRS,
    ],
    DataGroups.AUDIO: [
        SequenceLayout.HEAD_AUDIO_VRS,
        SequenceLayout.OBSERVER_AUDIO_VRS,
    ],
    DataGroups.EYE_TRACKING: [
        SequenceLayout.HEAD_ET_VRS,
        SequenceLayout.HEAD_EYE_GAZE_DIR,
        SequenceLayout.OBSERVER_ET_VRS,
        SequenceLayout.OBSERVER_EYE_GAZE_DIR,
    ],
    DataGroups.OBJECT_BOUNDING_BOX: [
        SequenceLayout.OBJECTS_BOXY_DIR,
    ],
    DataGroups.OBJECT_MESH: [
        SequenceLayout.OBJECTS_SHAPER_DIR,
    ],
    DataGroups.NARRATION: [
        SequenceLayout.NARRATION_DIR,
    ],
}


def get_group_definitions() -> dict[str, list[str]]:
    """Map each group name to the sequence-relative paths it contains.

    Mirrors the ``data_groups`` block of the release sequence config.
    """
    return {dg.value: list(files) for dg, files in _GROUP_FILES.items()}
