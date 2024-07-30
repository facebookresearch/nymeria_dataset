# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from zipfile import is_zipfile, ZipFile

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from .definitions import DataGroups, NYMERIA


class DlConfig(Enum):
    CHUCK_SIZE_BYTE = 8192
    READ_BYTE = 4096
    RETRY = 5
    BACKOFF_FACTOR = 3


class DlStatus(Enum):
    UNKNOWN = None
    SUCCESS = "success"
    IGNORED = "ignored, file already downloaded"
    WARN_NOTFOUND = "warning, missing download link"
    ERR_SHA1SUM = "error, sha1sum mismatch"
    ERR_MEMORY = "error, insufficient disk space"
    ERR_NETWORK = "error, network"


@dataclass
class DlLink:
    filename: str
    sha1sum: str
    file_size_bytes: int
    download_url: str

    data_group: DataGroups
    status: DlStatus = DlStatus.UNKNOWN

    def __post_init__(self):
        prefix = f"{NYMERIA.name.value}_v{NYMERIA.version.value}_"
        if prefix not in self.filename:
            self.status = (
                f"Version mismatch with the stable release v{NYMERIA.version.value}. "
                f"Please download the latest cdn json from {NYMERIA.website.value}"
            )
            raise ValueError(self.status)
        self.filename = self.filename.replace(prefix, "")

    @property
    def seq_name(self) -> str:
        return "_".join(self.filename.split("_")[0:6])

    @property
    def logdir(self) -> str:
        return "logs"

    def __check_sha1sum(self, file_path: Path) -> None:
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while chunk := f.read(DlConfig.READ_BYTE.value):
                sha1.update(chunk)

        computed = sha1.hexdigest()
        if self.sha1sum != computed:
            self.status = DlStatus.ERR_SHA1SUM
            raise RuntimeError(
                f"sha1sum mismatch, computed {computed}, expected {self.sha1sum}"
            )

    def __check_outdir(self, outdir: Path) -> None:
        assert (
            outdir.name == self.seq_name
        ), f"Output directory name ({outdir.name}) mismatch with sequence {self.seq_name}"
        outdir.mkdir(exist_ok=True)

    def get(self, outdir: Path, ignore_existing: bool = True) -> None:
        """This function throws error if not successful"""
        flag = outdir / self.logdir / self.data_group.name
        if flag.is_file() and ignore_existing:
            self.status = DlStatus.IGNORED
            return

        self.__check_outdir(outdir)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_filename = Path(tmpdir) / self.filename
            print(f"Donwload {self.filename} -> {tmp_filename}")

            # download with streaming
            session = requests.Session()
            retries = Retry(
                total=DlConfig.RETRY.value,
                backoff_factor=DlConfig.BACKOFF_FACTOR.value,
                # todo add comment
                status_forcelist=[429, 500, 502, 503, 504],
            )

            session.mount("https://", HTTPAdapter(max_retries=retries))
            with session.get(self.download_url, stream=True) as r:

                free_outdir = shutil.disk_usage(outdir).free
                free_tmpdir = shutil.disk_usage(tmpdir).free
                if (
                    free_outdir < self.file_size_bytes
                    or free_tmpdir < self.file_size_bytes
                ):
                    self.status = DlStatus.ERR_MEMORY
                    raise RuntimeError(
                        "Insufficient disk space. "
                        f"Require {self.file_size_bytes}B, "
                        f"tmpdir availabel {free_tmpdir}B, outdir available {free_outdir}B"
                    )

                with open(tmp_filename, "wb") as f:
                    progress_bar = tqdm(
                        total=self.file_size_bytes, unit="iB", unit_scale=True
                    )
                    for chunk in r.iter_content(
                        chunk_size=DlConfig.CHUCK_SIZE_BYTE.value
                    ):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                    progress_bar.close()

                try:
                    r.raise_for_status()
                except Exception as e:
                    self.status = DlStatus.ERR_NETWORK
                    raise RuntimeError(e)

            # validate checksum
            print("validate sha1sum")
            self.__check_sha1sum(tmp_filename)

            # move from tmp -> dst
            if is_zipfile(tmp_filename):
                print("unzip")
                with ZipFile(tmp_filename) as zf:
                    zf.extractall(outdir)
            else:
                dst_file = outdir / self.data_group.value
                dst_file.parent.mkdir(exist_ok=True)
                shutil.move(src=tmp_filename, dst=dst_file)

        print(f"Download {self.filename} -> {outdir}")
        self.status = DlStatus.SUCCESS

        # create a flag
        flag.parent.mkdir(exist_ok=True)
        flag.touch()


class DownloadManager:
    def __init__(self, cdn_json: Path, out_rootdir: Path):
        self.cdn_json = cdn_json
        assert self.cdn_json.is_file(), f"{self.cdn_json} not found"

        self.out_rootdir = out_rootdir
        self.out_rootdir.mkdir(exist_ok=True)

        with open(self.cdn_json, "r") as f:
            data = json.load(f)
            self._sequences = data.get("sequences", {})
            assert len(
                self._sequences
            ), "No sequence found. Please check the json file is correct."
        self._logs = {}

    @property
    def sequences(self) -> Dict[str, Any]:
        return self._sequences

    @property
    def logfile(self) -> Path:
        return self.out_rootdir / "download_summary"

    def __prepare(
        self,
        match_key: str,
        selected_groups: List["DataGroups"],
    ) -> Set["DataGroups"]:
        # FIXME modify the code once metadata.json is fixed
        # selected_groups += [DataGroups.license, DataGroups.metadata_json]
        selected_groups += [DataGroups.LICENSE]
        selected_groups = set(selected_groups)

        num_seqs = 0
        total_gb = 0
        self._logs = {}

        summary = {x.name: 0 for x in DataGroups}
        for seq, dgs in self.sequences.items():
            if match_key not in seq:
                continue

            num_seqs += 1
            self._logs[seq] = {}
            for dg in selected_groups:
                if dg.name not in dgs:
                    self._logs[seq][dg.name] = DlStatus.WARN_NOTFOUND.value
                    summary[dg.name] += 1
                    continue
                else:
                    self._logs[seq][dg.name] = None
                    dl = DlLink(**{**dgs.get(dg.name, {}), "data_group": dg})
                    total_gb += dl.file_size_bytes / (2**30)

        # populate confirmation msg
        msg = "\t" + "\n\t".join([x.value for x in selected_groups])
        free_disk_gb = shutil.disk_usage(self.out_rootdir).free / (2**30)
        confirm = (
            input(
                f"Download summary\n"
                f"  Output rootdir: {self.out_rootdir}\n"
                f"  Number sequences: {num_seqs}\n"
                f"  Total memory (GB): {total_gb}\n"
                f"  Available free disk space (GB): {free_disk_gb}\n"
                f"  Selected data groups:\n{msg}\n"
                f"Proceed: [y/n] "
            ).lower()
            == "y"
        )
        if not confirm:
            # save log which populates the list of files for downloading
            self.__logging(summary=summary)
            exit(1)
        return selected_groups

    def __logging(
        self, dl: Optional[DlLink] = None, summary: Optional[Dict[str, Any]] = None
    ) -> None:
        if dl is not None:
            self._logs[dl.seq_name][dl.data_group.name] = dl.status.value
        if summary is not None:
            self._logs["summary"] = summary
        with open(self.logfile, "w") as f:
            json.dump(self._logs, f, indent=2)

        if summary is not None:
            print(f"save {self.logfile}")

    def download(
        self,
        match_key: str,
        selected_groups: List["DataGroups"],
        ignore_existing: bool = True,
    ) -> None:
        selected_groups = self.__prepare(match_key, selected_groups)
        self.__logging()

        summary = {x.name: 0 for x in DlStatus}
        for seq_name, dgs in self.sequences.items():
            if match_key not in seq_name:
                continue

            outdir = self.out_rootdir / seq_name
            for dg in selected_groups:
                if dg.name not in dgs:
                    continue

                dl = DlLink(**{**dgs[dg.name], "data_group": dg})
                try:
                    dl.get(outdir, ignore_existing=ignore_existing)
                except Exception as e:
                    print(f"downloading failure:, {e}")

                summary[dl.status.name] += 1
                self.__logging(dl)

        self.__logging(summary)
        print(f"Dataset download to {self.out_rootdir}")
        print(f"Brief download summary: {json.dumps(summary, indent=2)}")
        print(f"Detailed summary saved to {self.logfile}")
