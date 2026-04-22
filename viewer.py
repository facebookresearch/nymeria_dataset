# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import click
import rerun as rr
from loguru import logger
from nymeria.data_provider import NymeriaDataProvider
from nymeria.data_viewer import NymeriaViewer
from nymeria.definitions import BodyModel


@click.command()
@click.option(
    "-i", "sequence_dir", type=Path, required=True, help="The directory of sequence "
)
@click.option(
    "-s", "save_rrd", is_flag=True, default=False, help="Save rerun into logfile"
)
@click.option(
    "--body-model",
    "body_model",
    type=click.Choice(["momentum", "smpl", "mhr"], case_sensitive=False),
    default="momentum",
    help="Body model to use: momentum (default), smpl, or mhr",
)
@click.option(
    "--smpl-model-path",
    "smpl_model_path",
    type=Path,
    default=None,
    help="Path to SMPL model .pkl file (required when --body-model=smpl)",
)
def main(
    sequence_dir: Path, save_rrd: bool, body_model: str, smpl_model_path: Path
) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<level>{level: <7}</level> <blue>{name}.py:</blue><green>{function}</green><yellow>:{line}</yellow> {message}",
        level="INFO",
    )

    if body_model == "smpl" and smpl_model_path is None:
        raise click.UsageError("--smpl-model-path is required when --body-model=smpl")

    # See NymeriaDataProviderConfig for configuration
    nymeria_dp = NymeriaDataProvider(
        sequence_rootdir=sequence_dir,
        load_wrist=True,
        body_model=BodyModel(body_model),
        smpl_model_path=str(smpl_model_path) if smpl_model_path else None,
    )

    output_rrd: Path = sequence_dir / "nymeria.rrd" if save_rrd else None
    viewer = NymeriaViewer(output_rrd=output_rrd)
    viewer(nymeria_dp)
    if save_rrd:
        logger.info(f"Save visualization to {output_rrd=}")

    rr.disconnect()


if __name__ == "__main__":
    main()
