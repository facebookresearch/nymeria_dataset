# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download NymeriaPlus sequences from signed URL JSON files."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_CHUNK_SIZE_BYTES = 8192
_RETRY_COUNT = 5
_BACKOFF_FACTOR = 3
_IGNORED_GROUPS = {"video_main_rgb"}
_LICENSE_KEY = "LICENSE"
_DEFAULT_NUM_WORKERS = 4


class DownloadStatus(Enum):
    """Per-artifact download outcome."""

    UNKNOWN = "unknown"
    SUCCESS = "success"
    IGNORED = "ignored, file already downloaded"
    LICENSE_COPIED = "copied local NYMERIAPLUS_DATASET_LICENSE"
    ERR_SHA1SUM = "error, sha1sum mismatch"
    ERR_MEMORY = "error, insufficient disk space"
    ERR_NETWORK = "error, network"
    ERR_EXTRACT = "error, zip extraction"
    ERR_DESTINATION = "error, destination path"


@dataclass(frozen=True)
class DownloadPlan:
    """Dry summary of the work implied by one URL JSON."""

    num_sequences: int
    num_artifacts: int
    num_ignored_artifacts: int
    total_size_bytes: int
    num_license_downloads: int
    num_license_copies: int

    @property
    def total_size_gib(self) -> float:
        return self.total_size_bytes / (2**30)


@dataclass
class DownloadLink:
    """One downloadable artifact for one sequence."""

    key: str
    filename: str
    sha1sum: str
    file_size_bytes: int
    download_url: str
    status: DownloadStatus = DownloadStatus.UNKNOWN

    @classmethod
    def from_json(cls, key: str, data: dict[str, Any]) -> "DownloadLink":
        missing = {
            field
            for field in ("filename", "sha1sum", "file_size_bytes", "download_url")
            if field not in data
        }
        if missing:
            raise ValueError(f"{key} missing required fields: {sorted(missing)}")
        return cls(
            key=key,
            filename=str(data["filename"]),
            sha1sum=str(data["sha1sum"]),
            file_size_bytes=int(data["file_size_bytes"]),
            download_url=str(data["download_url"]),
        )

    @property
    def is_zip(self) -> bool:
        return self.filename.endswith(".zip")

    def get(
        self,
        sequence_dir: Path,
        *,
        destination: Path | None,
        flag_path: Path,
        ignore_existing: bool = True,
    ) -> DownloadStatus:
        """Download, verify, and install this artifact."""

        if flag_path.is_file() and ignore_existing:
            self.status = DownloadStatus.IGNORED
            return self.status

        sequence_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            tmp_file = tmpdir / self.filename
            logger.info("download %s", self.filename)
            self._check_disk_space(sequence_dir, tmpdir)
            self._download_to_file(tmp_file)
            self._install_tmp_file(tmp_file, sequence_dir, destination)

        self.status = DownloadStatus.SUCCESS
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.touch()
        return self.status

    def _check_disk_space(self, sequence_dir: Path, tmpdir: Path) -> None:
        free_sequence = shutil.disk_usage(sequence_dir).free
        free_tmp = shutil.disk_usage(tmpdir).free
        if free_sequence >= self.file_size_bytes and free_tmp >= self.file_size_bytes:
            return
        self.status = DownloadStatus.ERR_MEMORY
        raise RuntimeError(
            "insufficient disk space: "
            f"need {self.file_size_bytes} bytes, "
            f"tmp has {free_tmp}, output has {free_sequence}"
        )

    def _download_to_file(self, tmp_file: Path) -> None:
        sha1 = hashlib.sha1()
        session = _requests_session()
        try:
            with session.get(self.download_url, stream=True) as response:
                with tmp_file.open("wb") as f:
                    progress = tqdm(
                        total=self.file_size_bytes,
                        unit="iB",
                        unit_scale=True,
                        desc=self.filename,
                    )
                    try:
                        for chunk in response.iter_content(
                            chunk_size=_CHUNK_SIZE_BYTES
                        ):
                            if not chunk:
                                continue
                            f.write(chunk)
                            sha1.update(chunk)
                            progress.update(len(chunk))
                    finally:
                        progress.close()
                response.raise_for_status()
        except Exception as e:
            self.status = DownloadStatus.ERR_NETWORK
            raise RuntimeError(e) from e

        computed = sha1.hexdigest()
        if computed == self.sha1sum:
            return
        self.status = DownloadStatus.ERR_SHA1SUM
        raise RuntimeError(
            f"sha1sum mismatch for {self.filename}: "
            f"computed {computed}, expected {self.sha1sum}"
        )

    def _install_tmp_file(
        self,
        tmp_file: Path,
        sequence_dir: Path,
        destination: Path | None,
    ) -> None:
        if self.is_zip:
            try:
                _extract_zip(tmp_file, sequence_dir)
            except Exception as e:
                self.status = DownloadStatus.ERR_EXTRACT
                raise RuntimeError(e) from e
            return

        if destination is None:
            self.status = DownloadStatus.ERR_DESTINATION
            raise RuntimeError(f"no destination path for {self.key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_file), str(destination))


@dataclass
class _Job:
    """One unit of parallel work for the download pool.

    A job either downloads a single artifact (``link`` set) or copies the
    bundled license into a sequence directory (``link`` is ``None``).
    """

    seq_name: str
    key: str
    sequence_dir: Path
    flag_path: Path
    link: DownloadLink | None = None


class DownloadManager:
    """Download every sequence artifact listed in a NymeriaPlus URL JSON."""

    def __init__(
        self,
        url_json: Path,
        out_rootdir: Path,
        *,
        license_source: Path | None = None,
        select: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.url_json = Path(url_json)
        if not self.url_json.is_file():
            raise FileNotFoundError(f"URL JSON not found: {self.url_json}")

        self.out_rootdir = Path(out_rootdir)
        self.out_rootdir.mkdir(parents=True, exist_ok=True)
        self.license_source = (
            Path(license_source)
            if license_source is not None
            else Path(__file__).resolve().parents[1] / "NYMERIAPLUS_DATASET_LICENSE"
        )

        with self.url_json.open("r") as f:
            data = json.load(f)
        self.sequences = data.get("sequences")
        self.sequence_config = data.get("sequence_config")
        if not isinstance(self.sequences, dict) or not self.sequences:
            raise ValueError("input JSON must contain a non-empty `sequences` object")
        if not isinstance(self.sequence_config, dict):
            raise ValueError("input JSON must contain a `sequence_config` object")

        self.sequences = self._select_sequences(self.sequences, select)

        self._single_file_destinations = self._build_single_file_destinations()
        self._logs: dict[str, dict[str, str | None]] = {}

    @property
    def logfile(self) -> Path:
        return self.out_rootdir / "download_summary.json"

    @property
    def data_summary_file(self) -> Path:
        return self.out_rootdir / "data_summary.json"

    @property
    def flag_rootdir(self) -> Path:
        return self.out_rootdir / ".download_logs"

    def build_plan(self) -> DownloadPlan:
        num_artifacts = 0
        num_ignored = 0
        total_size = 0
        license_downloads = 0
        license_copies = 0

        for _seq_name, entries in self.sequences.items():
            if _LICENSE_KEY not in entries:
                license_copies += 1
            for key, data in entries.items():
                if key in _IGNORED_GROUPS:
                    num_ignored += 1
                    continue
                if key == _LICENSE_KEY:
                    license_downloads += 1
                num_artifacts += 1
                total_size += int(data.get("file_size_bytes", 0))

        return DownloadPlan(
            num_sequences=len(self.sequences),
            num_artifacts=num_artifacts,
            num_ignored_artifacts=num_ignored,
            total_size_bytes=total_size,
            num_license_downloads=license_downloads,
            num_license_copies=license_copies,
        )

    def write_data_summary(self) -> None:
        available_keys = sorted(
            {key for entries in self.sequences.values() for key in entries.keys()}
        )
        missing_license = [
            seq_name
            for seq_name, entries in self.sequences.items()
            if _LICENSE_KEY not in entries
        ]
        with self.data_summary_file.open("w") as f:
            json.dump(
                {
                    "available_sequences": list(self.sequences.keys()),
                    "available_artifact_keys": available_keys,
                    "ignored_artifact_keys": sorted(_IGNORED_GROUPS),
                    "missing_license_sequences": missing_license,
                    "sequence_config": self.sequence_config,
                },
                f,
                indent=2,
            )

    def download(
        self,
        *,
        ignore_existing: bool = True,
        num_workers: int = _DEFAULT_NUM_WORKERS,
    ) -> dict[str, int]:
        self.write_data_summary()
        self._logs = {seq_name: {} for seq_name in self.sequences}
        summary = {status.name: 0 for status in DownloadStatus}

        jobs = self._collect_jobs()
        num_workers = max(1, num_workers)
        logger.info("running %d download jobs with %d workers", len(jobs), num_workers)

        lock = threading.Lock()

        def handle(job: _Job) -> None:
            status = self._run_job(job, ignore_existing)
            with lock:
                summary[status.name] += 1
                self._logs[job.seq_name][job.key] = status.value
                self._write_download_summary(summary)

        if num_workers == 1:
            for job in jobs:
                handle(job)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(handle, job) for job in jobs]
                for future in as_completed(futures):
                    future.result()

        self._write_download_summary(summary)
        return summary

    def _collect_jobs(self) -> list[_Job]:
        """Flatten every sequence into an independent list of download jobs."""

        jobs: list[_Job] = []
        for seq_name, entries in self.sequences.items():
            sequence_dir = self.out_rootdir / seq_name
            for key, data in entries.items():
                if key in _IGNORED_GROUPS:
                    continue
                jobs.append(
                    _Job(
                        seq_name=seq_name,
                        key=key,
                        sequence_dir=sequence_dir,
                        flag_path=self._flag_path(seq_name, key),
                        link=DownloadLink.from_json(key, data),
                    )
                )
            if _LICENSE_KEY not in entries:
                jobs.append(
                    _Job(
                        seq_name=seq_name,
                        key=_LICENSE_KEY,
                        sequence_dir=sequence_dir,
                        flag_path=self._flag_path(seq_name, _LICENSE_KEY),
                        link=None,
                    )
                )
        return jobs

    def _run_job(self, job: _Job, ignore_existing: bool) -> DownloadStatus:
        """Execute one job and return its outcome; never raises."""

        if job.link is None:
            try:
                return self._copy_license(
                    job.sequence_dir, job.flag_path, ignore_existing
                )
            except Exception as e:
                logger.error("failed to copy local license for %s: %s", job.seq_name, e)
                return DownloadStatus.ERR_DESTINATION

        try:
            destination = self._resolve_destination(job.sequence_dir, job.link)
            return job.link.get(
                job.sequence_dir,
                destination=destination,
                flag_path=job.flag_path,
                ignore_existing=ignore_existing,
            )
        except Exception as e:
            status = job.link.status
            if status == DownloadStatus.UNKNOWN:
                status = DownloadStatus.ERR_NETWORK
            logger.error("failed to download %s/%s: %s", job.seq_name, job.key, e)
            return status

    @staticmethod
    def _select_sequences(
        sequences: dict[str, Any],
        select: list[str] | tuple[str, ...] | None,
    ) -> dict[str, Any]:
        """Keep sequences whose name contains any of the `select` substrings.

        An empty or omitted `select` keeps every sequence. A selection that
        matches no sequence is an error, since it would download nothing.
        """
        if not select:
            return sequences

        patterns = [p for p in select if p]
        if not patterns:
            return sequences

        selected = {
            seq_name: entries
            for seq_name, entries in sequences.items()
            if any(pattern in seq_name for pattern in patterns)
        }
        if not selected:
            raise ValueError(
                "no sequences matched the requested selection "
                f"{sorted(patterns)}; available sequences: {len(sequences)}"
            )
        logger.info(
            "selected %d of %d sequences matching %s",
            len(selected),
            len(sequences),
            sorted(patterns),
        )
        return selected

    def _build_single_file_destinations(self) -> dict[str, str]:
        ungrouped = self.sequence_config.get("ungrouped_files")
        if not isinstance(ungrouped, list):
            raise ValueError("sequence_config must contain `ungrouped_files`")

        destinations = {_path_to_key(str(rel)): str(rel) for rel in ungrouped}
        destinations[_LICENSE_KEY] = "LICENSE"
        return destinations

    def _resolve_destination(
        self, sequence_dir: Path, link: DownloadLink
    ) -> Path | None:
        if link.is_zip:
            return None
        rel = self._single_file_destinations.get(link.key)
        if rel is None:
            raise ValueError(
                f"no destination for non-zip artifact `{link.key}` in sequence_config"
            )
        return sequence_dir / rel

    def _copy_license(
        self,
        sequence_dir: Path,
        flag_path: Path,
        ignore_existing: bool,
    ) -> DownloadStatus:
        if flag_path.is_file() and ignore_existing:
            return DownloadStatus.IGNORED
        destination = sequence_dir / "LICENSE"
        if not self.license_source.is_file():
            raise FileNotFoundError(
                f"local NymeriaPlus dataset license not found: {self.license_source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.license_source, destination)
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.touch()
        return DownloadStatus.LICENSE_COPIED

    def _flag_path(self, seq_name: str, key: str) -> Path:
        return self.flag_rootdir / seq_name / key

    def _write_download_summary(self, summary: dict[str, int]) -> None:
        with self.logfile.open("w") as f:
            json.dump(
                {
                    "sequences": self._logs,
                    "download_summary": summary,
                },
                f,
                indent=2,
            )


def _path_to_key(path: str) -> str:
    return path.replace("/", "_").replace(".", "_")


def _requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=_RETRY_COUNT,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


def _extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not _is_relative_to(target, destination):
                raise RuntimeError(f"zip member escapes destination: {member.filename}")
        zf.extractall(destination)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
