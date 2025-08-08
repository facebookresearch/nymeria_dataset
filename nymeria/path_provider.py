# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from pathlib import Path

from nymeria.definitions import BodyFiles, MetaFiles, Subpaths, TextFiles


class SequencePathProvider:
    """
    \brief Each sequence contains the following subfolders:
        - recording_head
        - recording_lwrist
        - recording_rwrist
        - recording_observer
        - body
        - narration
    """

    def __init__(self, rootdir: Path) -> None:
        assert rootdir.is_dir(), f"{rootdir=} not found"
        self.rootdir = rootdir
        self.license = rootdir / MetaFiles.license
        self.metadata = rootdir / MetaFiles.metadata_json
        self.body_paths = BodyFiles(
            **{
                f.name: str(rootdir / getattr(BodyFiles, f.name))
                for f in fields(BodyFiles)
            }
        )
        self.narration_paths = TextFiles(
            **{
                f.name: str(rootdir / getattr(TextFiles, f.name))
                for f in fields(TextFiles)
            }
        )

        self.recording_head: Path = rootdir / Subpaths.recording_head
        self.recording_lwrist: Path = rootdir / Subpaths.recording_lwrist
        self.recording_rwrist: Path = rootdir / Subpaths.recording_rwrist
        self.recording_observer: Path = rootdir / Subpaths.recording_observer

    def __repr__(self) -> str:
        return (
            f"SequencePaths(\n"
            f"  license={self.license},\n"
            f"  metadata={self.metadata},\n"
            f"  body_files={self.body_files},\n"
            f"  recording_head={self.recording_head},\n"
            f"  recording_observer={self.recording_observer},\n"
            f"  recording_lwrist={self.recording_lwrist},\n"
            f"  recording_rwrist={self.recording_rwrist}\n)"
        )
