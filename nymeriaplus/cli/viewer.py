# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI: load a NymeriaPlus sequence, synchronize, and open the viewer."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from nymeriaplus._logging import configure as configure_logging
from nymeriaplus.data_loader import NymeriaPlusDataLoader
from nymeriaplus.rendering import launch

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "sequence_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="NymeriaPlus sequence root directory.",
)
@click.option(
    "--smpl-model-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a SMPL .pkl model file. Required to visualize the SMPL mesh.",
)
@click.option(
    "--target-fps",
    type=float,
    default=30.0,
    show_default=True,
    help="Synchronization target frame rate.",
)
@click.option(
    "--ui-scale",
    type=click.FloatRange(min=0.25, max=4.0, min_open=True),
    default=None,
    show_default="auto",
    help="Override ImGui UI scale. By default, the viewer uses the window content scale.",
)
def main(
    sequence_dir: Path,
    smpl_model_path: Path | None,
    target_fps: float,
    ui_scale: float | None,
) -> None:
    configure_logging()

    loader = NymeriaPlusDataLoader(
        sequence_dir,
        load_recordings=True,
        load_smpl=True,
        load_mhr=True,
        load_xsens=True,
        load_bbox=True,
        load_mesh=True,
        smpl_model_path=smpl_model_path,
    )
    synced = loader.synchronize_data(target_fps=target_fps)

    smpl_loader = loader.smpl_body if smpl_model_path is not None else None
    launch(loader, synced, smpl_loader, loader.mhr_body, ui_scale=ui_scale)


if __name__ == "__main__":
    main()
