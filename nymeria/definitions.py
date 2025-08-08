# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, fields
from enum import Enum

"""
Each sequence folder follows the following structure.
Files might be missing if not downloaded. 
  ├── LICENSE 
  ├── metadata.json
  ├── body
  │   ├── xdata_blueman.glb
  │   ├── xdata.healthcheck
  │   ├── xdata.mvnx
  │   └── xdata.npz
  ├── narration
  │   ├── activity_summarization.csv
  │   ├── atomic_action.csv
  │   └── motion_narration.csv
  ├── recording_head / recording_observer
  │   ├── data
  │   │   ├── data.vrs
  │   │   ├── et.vrs
  │   │   └── motion.vrs
  │   └── mps
  │       ├── eye_gaze
  │       │   ├── general_eye_gaze.csv
  │       │   └── personalized_eye_gaze.csv
  │       └── slam
  │           ├── closed_loop_trajectory.csv
  │           ├── online_calibration.jsonl
  │           ├── open_loop_trajectory.csv
  │           ├── semidense_observations.csv.gz
  │           ├── semidense_points.csv.gz
  │           └── summary.json
  └── recording_rwrist / recording_lwrist
      ├── data
      │   ├── data.vrs
      │   └── motion.vrs
      └── mps
          └── slam
              ├── closed_loop_trajectory.csv
              ├── online_calibration.jsonl
              ├── open_loop_trajectory.csv
              ├── semidense_observations.csv.gz
              ├── semidense_points.csv.gz
              └── summary.json

"""

NYMERIA_VERSION: str = "v0.0"


@dataclass(frozen=True)
class MetaFiles:
    license: str = "LICENSE"
    metadata_json: str = "metadata.json"


@dataclass(frozen=True)
class Subpaths(MetaFiles):
    body: str = "body"
    text: str = "narration"

    recording_head: str = "recording_head"
    recording_lwrist: str = "recording_lwrist"
    recording_rwrist: str = "recording_rwrist"
    recording_observer: str = "recording_observer"

    vrs: str = "data"
    mps: str = "mps"
    mps_slam: str = "mps/slam"
    mps_gaze: str = "mps/eye_gaze"


@dataclass(frozen=True)
class BodyFiles:
    xsens_processed: str = f"{Subpaths.body}/xdata.npz"
    xsens_raw: str = f"{Subpaths.body}/xdata.mvnx"
    momentum_model: str = f"{Subpaths.body}/xdata_blueman.glb"


@dataclass(frozen=True)
class TextFiles:
    motion_narration: str = f"{Subpaths.text}/motion_narration.csv"
    atomic_action: str = f"{Subpaths.text}/atomic_action.csv"
    activity_summarization: str = f"{Subpaths.text}/activity_summarization.csv"


@dataclass(frozen=True)
class VrsFiles:
    data: str = f"{Subpaths.vrs}/data.vrs"
    motion: str = f"{Subpaths.vrs}/motion.vrs"
    et: str = f"{Subpaths.vrs}/et.vrs"


@dataclass(frozen=True)
class SlamFiles:
    closed_loop_trajectory: str = f"{Subpaths.mps_slam}/closed_loop_trajectory.csv"
    online_calibration: str = f"{Subpaths.mps_slam}/online_calibration.jsonl"
    open_loop_trajectory: str = f"{Subpaths.mps_slam}/open_loop_trajectory.csv"
    semidense_points: str = f"{Subpaths.mps_slam}/semidense_points.csv.gz"
    semidense_observations: str = f"{Subpaths.mps_slam}/semidense_observations.csv.gz"
    location_summary: str = f"{Subpaths.mps_slam}/summary.json"


@dataclass(frozen=True)
class GazeFiles:
    general_gaze: str = f"{Subpaths.mps_gaze}/general_eye_gaze.csv"
    personalized_gaze: str = f"{Subpaths.mps_gaze}/personalized_eye_gaze.csv"


class DataGroups(Enum):
    """
    \brief Each variable defines one atomic downloadable element
    """

    LICENSE = Subpaths.license
    metadata_json = Subpaths.metadata_json

    body_motion = Subpaths.body

    recording_head = Subpaths.recording_head
    recording_head_data_data_vrs = f"{Subpaths.recording_head}/{VrsFiles.data}"
    recording_lwrist = Subpaths.recording_lwrist
    recording_rwrist = Subpaths.recording_rwrist
    recording_observer = Subpaths.recording_observer
    recording_observer_data_data_vrs = f"{Subpaths.recording_observer}/{VrsFiles.data}"

    narration_motion_narration_csv = TextFiles.motion_narration
    narration_atomic_action_csv = TextFiles.atomic_action
    narration_activity_summarization_csv = TextFiles.activity_summarization

    semidense_observations = "semidense_observations"

    body_xdata_mvnx = "body_xdata_mvnx"


def get_group_definitions() -> dict[str, list]:
    """
    \brief Definition of DataGroups
           File paths are relative with respect to each sequence folder.
           Some sequences might missing certain files/data groups
           due to errors occurred from data collection or processing.
           There is one url per data group per sequence.
           Data groups with multiple files are packed into zip files.
    """
    AriaFiles = (
        [f.default for f in fields(VrsFiles) if "data" not in f.name]
        + [f.default for f in fields(SlamFiles) if "observations" not in f.name]
        + [f.default for f in fields(GazeFiles)]
    )
    miniAriaFiles = [f.default for f in fields(VrsFiles) if "et" not in f.name] + [
        f.default for f in fields(SlamFiles) if "observations" not in f.name
    ]

    g_defs = {x.name: [x.value] for x in DataGroups}
    g_defs[DataGroups.body_motion.name] = [x.default for x in fields(BodyFiles)]

    g_defs[DataGroups.body_xdata_mvnx.name] = [BodyFiles.xsens_raw]

    for x in [DataGroups.recording_head, DataGroups.recording_observer]:
        g_defs[x.name] = [f"{x.name}/{f}" for f in AriaFiles]

    for x in [DataGroups.recording_rwrist, DataGroups.recording_lwrist]:
        g_defs[x.name] = [f"{x.name}/{f}" for f in miniAriaFiles]

    g_defs[DataGroups.semidense_observations.name] = []
    for x in fields(Subpaths):
        if "recording" in x.name:
            g_defs[DataGroups.semidense_observations.name].append(
                f"{x.default}/{SlamFiles.semidense_observations}"
            )

    print("=== group definitions (group_name: [group_files]) ===")
    print(json.dumps(g_defs, indent=2))

    return g_defs


# get_group_definitions()
