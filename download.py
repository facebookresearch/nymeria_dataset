# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import click
from loguru import logger
from nymeria.definitions import DataGroups
from nymeria.download_utils import DownloadManager


def get_groups(full: bool = False) -> list[DataGroups]:
    """
    By default all data present in nymeria_download_urls.json will be downloaded.
    For selective download, comment out lines to disable certain groups.
    See nymeria/definitions.py GroupDefs for the files included by each group.
    
    For MVNX-only downloads using nymeria_download_urls_xdata_mvnx.json:
    - Comment out all groups except body_xdata_mvnx
    - Or use only [DataGroups.body_xdata_mvnx] to download MVNX files only
    """
    return [
        # DataGroups.LICENSE,
        # DataGroups.metadata_json,
        # DataGroups.body_motion,  # Contains all body files (npz, mvnx, glb)
        # DataGroups.recording_head,
        # DataGroups.recording_head_data_data_vrs,
        # DataGroups.recording_lwrist,
        # DataGroups.recording_rwrist,
        # DataGroups.recording_observer,
        # DataGroups.recording_observer_data_data_vrs,
        # DataGroups.narration_motion_narration_csv,
        # DataGroups.narration_atomic_action_csv,
        # DataGroups.narration_activity_summarization_csv,
        # DataGroups.semidense_observations,
        
        # For MVNX-only downloads with nymeria_download_urls_xdata_mvnx.json
        DataGroups.body_xdata_mvnx,  # Downloads only xdata.mvnx files
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
def main(url_json: Path, rootdir: Path, match_key: str = "2023") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<level>{level: <7}</level> <blue>{name}.py:</blue><green>{function}</green><yellow>:{line}</yellow> {message}",
        level="INFO",
    )

    dl = DownloadManager(url_json, out_rootdir=rootdir)
    dl.download(match_key=match_key, selected_groups=get_groups(), ignore_existing=True)


if __name__ == "__main__":
    main()
