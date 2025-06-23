# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class XSensConstants:
    """
    \brief Transformations segment_tXYZ and segment_qWXYZ are defined as
           from XSens segment/part coordinates to XSens world coordinates.
           See XSens manual for details.
    """

    num_parts: int = 23
    num_bones: int = 22
    num_sensors: int = 17
    k_timestamps_us: str = "timestamps_us"
    k_frame_count: str = "frameCount"
    k_framerate: str = "frameRate"
    k_part_tXYZ: str = "segment_tXYZ"
    k_part_qWXYZ: str = "segment_qWXYZ"
    k_ipose_part_tXYZ: str = "identity_segment_tXYZ"
    k_ipose_part_qWXYZ: str = "identity_segment_qWXYZ"
    k_tpose_part_tXYZ: str = "tpose_segment_tXYZ"
    k_tpose_part_qWXYZ: str = "tpose_segment_qWXYZ"
    k_foot_contacts: str = "foot_contacts"
    k_sensor_tXYZ: str = "sensor_tXYZ"
    k_sensor_qWXYZ: str = "sensor_qWXYZ"
    part_names = [
        "Pelvis",
        "L5",
        "L3",
        "T12",
        "T8",
        "Neck",
        "Head",
        "R_Shoulder",
        "R_UpperArm",
        "R_Forearm",
        "R_Hand",
        "L_Shoulder",
        "L_UpperArm",
        "L_Forearm",
        "L_Hand",
        "R_UpperLeg",
        "R_LowerLeg",
        "R_Foot",
        "R_Toe",
        "L_UpperLeg",
        "L_LowerLeg",
        "L_Foot",
        "L_Toe",
    ]  # num = 23
    kintree_parents: list[int] = [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        4,
        7,
        8,
        9,
        4,
        11,
        12,
        13,
        0,
        15,
        16,
        17,
        0,
        19,
        20,
        21,
    ]  # num = 23
    sensor_names: list[int] = [
        "Pelvis",
        "T8",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
    ]  # num = 17
