# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class NYMERIA(Enum):
    name = "Nymeria"
    version = "0.0"
    website = "https://www.projectaria.com/datasets/nymeria"


class DataGroups(Enum):
    LICENSE = "LICENSE"
    metadata_json = "metadata.json"

    body_motion = "body_motion"
    recording_head = "recording_head"
    recording_head_data_data_vrs = "recording_head/data/data.vrs"
    recording_lwrist = "recording_lwrist"
    recording_rwrist = "recording_rwrist"
    recording_observer = "recording_observer"
    recording_observer_data_data_vrs = "recording_observer/data/data.vrs"

    narration_motion_narration_csv = "narration/motion_narration.csv"
    narration_atomic_action_csv = "narration/atomic_action.csv"
    narration_activity_summarization_csv = "narration/activity_summarization.csv"

    semidense_observations = "semidense_observations"


class GroupDefs:
    """
    \brief Definition of DataGroups
           File paths are relative with respect to each sequence folder.
           Some sequences might missing certain files/data groups
           due to errors occured from data collection or processing.
           There is one url per data group per sequence.
           Data groups with multiple files are packed into zip files.
    """

    LICENSE = "LICENSE"
    metadata_json = "metadata.json"

    body_motion = ["body/xdata.npz", "body/xdata_blueman.glb", "body/data.mvnx"]

    recording_head = [
        "recording_head/data/motion.vrs",
        "recording_head/data/et.vrs",
        "recording_head/mps/slam/closed_loop_trajectory.csv",
        "recording_head/mps/slam/semidense_points.csv.gz",
        "recording_head/mps/slam/summary.json",
        "recording_head/mps/slam/online_calibration.jsonl",
        "recording_head/mps/eye_gaze",
    ]
    recording_head_data_data_vrs = "recording_head/data/data.vrs"

    recording_lwrist = [
        "recording_lwrist/data/motion.vrs",
        "recording_lwrist/mps/slam/closed_loop_trajectory.csv",
        "recording_lwrist/mps/slam/semidense_points.csv.gz",
        "recording_lwrist/mps/slam/summary.json",
        "recording_lwrist/mps/slam/online_calibration.jsonl",
    ]
    recording_rwrist = [
        "recording_rwrist/data/motion.vrs",
        "recording_rwrist/mps/slam/closed_loop_trajectory.csv",
        "recording_rwrist/mps/slam/semidense_points.csv.gz",
        "recording_rwrist/mps/slam/summary.json",
        "recording_rwrist/mps/slam/online_calibration.jsonl",
    ]

    recording_observer = [
        "recording_observer/data/motion.vrs",
        "recording_observer/data/et.vrs",
        "recording_observer/mps/slam/closed_loop_trajectory.csv",
        "recording_observer/mps/slam/semidense_points.csv.gz",
        "recording_observer/mps/slam/summary.json",
        "recording_observer/mps/slam/online_calibration.jsonl",
        "recording_observer/mps/eye_gaze",
    ]
    recording_observer_data_data_vrs = "recording_observer/data/data.vrs"

    semidense_observations = [
        "recording_head/mps/slam/semidense_observations.csv.gz",
        "recording_lwrist/mps/slam/semidense_observations.csv.gz",
        "recording_rwrist/mps/slam/semidense_observations.csv.gz",
        "recording_observer/mps/slam/semidense_observations.csv.gz",
    ]

    narration_motion_narration_csv = "narration/motion_narration.csv"
    narration_atomic_action_csv = "narration/atomic_action.csv"
    narration_activity_summarization_csv = "narration/activity_summarization.csv"
