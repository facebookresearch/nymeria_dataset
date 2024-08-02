# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class SequenceAttributes:
    date: str
    session_id: str
    fake_name: str
    act_id: str
    uid: str

    location: str
    script: str
    action_duration_sec: float = -1

    has_two_participants: bool = False
    pt2: str = None
    body_motion: bool = False

    head_data: bool = False
    head_slam: bool = False
    head_trajectory_m: float = None
    head_duration_sec: float = None
    head_general_gaze: bool = False
    head_personalized_gaze: bool = False

    left_wrist_data: bool = False
    left_wrist_slam: bool = False
    left_wrist_trajectory_m: float = None
    left_wrist_duration_sec: float = None

    right_wrist_data: bool = False
    right_wrist_slam: bool = False
    right_wrist_trajectory_m: float = None
    right_wrist_duration_sec: float = None

    observer_data: bool = False
    observer_slam: bool = False
    observer_general_gaze: bool = False
    observer_personalized_gaze: bool = False
    observer_trajectory_m: float = None
    observer_duration_sec: float = None

    timesync: bool = False

    motion_narration: bool = False
    atomic_action: bool = False
    activity_summarization: bool = False

    participant_gender: str = None
    participant_height_cm: float = -1
    participant_weight_kg: float = -1
    participant_bmi: float = -1
    participant_age_group: str = None
    participant_ethnicity: str = None
    participant_xsens_suit_size: str = None
