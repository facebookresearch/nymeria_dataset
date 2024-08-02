# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List

import click
from nymeria.definitions import DataGroups
from nymeria.download_utils import DownloadManager


def get_groups(full: bool = False) -> List[DataGroups]:
    """
    By default all data present in download_urls.JSON will be downloaded.
    For selective download, comment out lines to disable certain groups.
    See nymeria/definitions.py GroupDefs for the files included by each group.
    """
    return [
        DataGroups.LICENSE,
        DataGroups.metadata_json,
        DataGroups.body_motion,
        DataGroups.recording_head,
        DataGroups.recording_head_data_data_vrs,
        DataGroups.recording_lwrist,
        DataGroups.recording_rwrist,
        DataGroups.recording_observer,
        DataGroups.recording_observer_data_data_vrs,
        DataGroups.narration_motion_narration_csv,
        DataGroups.narration_atomic_action_csv,
        DataGroups.narration_activity_summarization_csv,
        DataGroups.semidense_observations,
    ]


@click.command()
@click.option(
    "-i",
    "url_json",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    required=True,
    help="The json file contains download urls. Follow README.md instructions to access this file.",
)
@click.option(
    "-o",
    "rootdir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=None,
    help="The root directory to hold the downloaded dataset",
)
@click.option(
    "-k",
    "match_key",
    default="2023",
    help="Partial key used to filter sequences for downloading"
    "Default key value = 2023, which include all available sequences",
)
def main(url_json: Path, rootdir: Path, match_key: str = "2023"):
    dl = DownloadManager(url_json, out_rootdir=rootdir)
    dl.download(match_key=match_key, selected_groups=get_groups(), ignore_existing=True)


if __name__ == "__main__":
    main()
