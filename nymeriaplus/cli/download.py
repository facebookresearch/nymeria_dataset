# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI for downloading NymeriaPlus sequences from signed URL JSON files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click
from nymeriaplus.downloader import DownloadManager


@click.command()
@click.option(
    "-i",
    "--input",
    "url_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON file containing NymeriaPlus signed download URLs.",
)
@click.option(
    "-o",
    "--output",
    "out_rootdir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    required=True,
    help="Output directory for downloaded sequence folders.",
)
@click.option(
    "-s",
    "--select",
    multiple=True,
    metavar="SUBSTRING",
    help=(
        "Only download sequences whose name contains this substring. "
        "Repeat to select multiple substrings (a sequence is downloaded if it "
        "matches any of them). Omit to download every sequence."
    ),
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Skip the confirmation prompt.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Redownload artifacts even when hidden resume markers already exist.",
)
@click.option(
    "-n",
    "--num-workers",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Number of artifacts to download in parallel.",
)
def main(
    url_json: Path,
    out_rootdir: Path,
    select: tuple[str, ...],
    yes: bool,
    overwrite: bool,
    num_workers: int,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manager = DownloadManager(url_json, out_rootdir, select=select)
    plan = manager.build_plan()
    free_gib = shutil.disk_usage(out_rootdir).free / (2**30)

    click.echo("Download summary")
    click.echo(f"  Input JSON: {url_json}")
    click.echo(f"  Output root: {out_rootdir}")
    if select:
        click.echo(f"  Selection substrings: {list(select)}")
    click.echo(f"  Sequences: {plan.num_sequences}")
    click.echo(f"  Artifacts to download: {plan.num_artifacts}")
    click.echo(f"  Ignored video_main_rgb artifacts: {plan.num_ignored_artifacts}")
    click.echo(f"  Total download size: {plan.total_size_gib:.2f} GiB")
    click.echo(f"  Available disk space: {free_gib:.2f} GiB")
    click.echo(f"  LICENSE downloads: {plan.num_license_downloads}")
    click.echo(f"  LICENSE local copies: {plan.num_license_copies}")
    click.echo(f"  Parallel workers: {num_workers}")

    if not yes and not click.confirm("Proceed?", default=False):
        raise click.Abort()

    summary = manager.download(ignore_existing=not overwrite, num_workers=num_workers)
    click.echo(f"Downloaded dataset to {out_rootdir}")
    click.echo(f"Detailed summary saved to {manager.logfile}")
    click.echo(f"Brief summary: {summary}")


if __name__ == "__main__":
    main()
