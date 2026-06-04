# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from nymeriaplus.downloader import DownloadManager, DownloadStatus


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return out.getvalue()


def _write_urls(path: Path, sequences: dict[str, dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "sequences": sequences,
                "sequence_config": {
                    "dataset_name": "NymeriaPlus",
                    "release": "v1.0",
                    "main": {
                        "recording": "recording_head/data/data.vrs",
                        "mps": "recording_head/mps",
                    },
                    "data_groups": {
                        "object_bounding_box": ["objects/boxy"],
                    },
                    "ungrouped_files": [
                        "LICENSE",
                        "metadata.json",
                        "recording_head/data/data.vrs",
                    ],
                },
            }
        )
    )


def _entry(filename: str, data: bytes, url: str) -> dict:
    return {
        "filename": filename,
        "sha1sum": _sha1(data),
        "file_size_bytes": len(data),
        "download_url": url,
    }


def test_download_extracts_zip_places_single_files_and_copies_license(
    tmp_path: Path,
    requests_mock,
) -> None:
    zip_data = _zip_bytes({"objects/boxy/3dbb.csv": b"bbox"})
    metadata = b'{"seq": true}\n'
    video = b"ignored"
    url_json = tmp_path / "urls.json"
    license_source = tmp_path / "NYMERIAPLUS_DATASET_LICENSE"
    out_root = tmp_path / "out"
    license_source.write_text("dataset license\n")
    _write_urls(
        url_json,
        {
            "seq0": {
                "object_bounding_box": _entry(
                    "NymeriaPlus_v1.0_seq0_object_bounding_box.zip",
                    zip_data,
                    "https://example.com/bbox.zip",
                ),
                "metadata_json": _entry(
                    "NymeriaPlus_v1.0_seq0_metadata.json",
                    metadata,
                    "https://example.com/metadata.json",
                ),
                "video_main_rgb": _entry(
                    "NymeriaPlus_v1.0_seq0_preview_rgb.mp4",
                    video,
                    "https://example.com/preview.mp4",
                ),
            },
        },
    )
    requests_mock.get("https://example.com/bbox.zip", content=zip_data)
    requests_mock.get("https://example.com/metadata.json", content=metadata)

    manager = DownloadManager(url_json, out_root, license_source=license_source)
    plan = manager.build_plan()
    assert plan.num_artifacts == 2
    assert plan.num_ignored_artifacts == 1
    assert plan.num_license_copies == 1

    summary = manager.download()

    assert summary[DownloadStatus.SUCCESS.name] == 2
    assert summary[DownloadStatus.LICENSE_COPIED.name] == 1
    assert (out_root / "seq0/objects/boxy/3dbb.csv").read_text() == "bbox"
    assert (out_root / "seq0/metadata.json").read_bytes() == metadata
    assert (out_root / "seq0/LICENSE").read_text() == "dataset license\n"
    assert not (out_root / "seq0/preview_rgb.mp4").exists()
    assert not (out_root / "seq0/logs").exists()
    assert (out_root / ".download_logs/seq0/object_bounding_box").is_file()
    assert (out_root / ".download_logs/seq0/metadata_json").is_file()
    assert (out_root / ".download_logs/seq0/LICENSE").is_file()
    assert not requests_mock.called or requests_mock.call_count == 2


def test_download_places_license_and_vrs_from_sequence_config(
    tmp_path: Path,
    requests_mock,
) -> None:
    license_data = b"downloaded license\n"
    vrs_data = b"vrs"
    url_json = tmp_path / "urls.json"
    out_root = tmp_path / "out"
    _write_urls(
        url_json,
        {
            "seq0": {
                "LICENSE": _entry(
                    "NymeriaPlus_v1.0_seq0_LICENSE",
                    license_data,
                    "https://example.com/license",
                ),
                "recording_head_data_data_vrs": _entry(
                    "NymeriaPlus_v1.0_seq0_recording_head_data_data.vrs",
                    vrs_data,
                    "https://example.com/data.vrs",
                ),
            },
        },
    )
    requests_mock.get("https://example.com/license", content=license_data)
    requests_mock.get("https://example.com/data.vrs", content=vrs_data)

    manager = DownloadManager(url_json, out_root)
    summary = manager.download()

    assert summary[DownloadStatus.SUCCESS.name] == 2
    assert (out_root / "seq0/LICENSE").read_bytes() == license_data
    assert (out_root / "seq0/recording_head/data/data.vrs").read_bytes() == vrs_data
    assert not (out_root / "seq0/logs").exists()
    assert (out_root / ".download_logs/seq0/LICENSE").is_file()
    assert (out_root / ".download_logs/seq0/recording_head_data_data_vrs").is_file()


def test_download_skips_completed_artifacts_from_hidden_markers(
    tmp_path: Path,
    requests_mock,
) -> None:
    metadata = b'{"seq": true}\n'
    url_json = tmp_path / "urls.json"
    license_source = tmp_path / "NYMERIAPLUS_DATASET_LICENSE"
    out_root = tmp_path / "out"
    license_source.write_text("dataset license\n")
    _write_urls(
        url_json,
        {
            "seq0": {
                "metadata_json": _entry(
                    "NymeriaPlus_v1.0_seq0_metadata.json",
                    metadata,
                    "https://example.com/metadata.json",
                ),
            },
        },
    )
    requests_mock.get("https://example.com/metadata.json", content=metadata)

    manager = DownloadManager(url_json, out_root, license_source=license_source)
    first_summary = manager.download()
    second_summary = manager.download()

    assert first_summary[DownloadStatus.SUCCESS.name] == 1
    assert first_summary[DownloadStatus.LICENSE_COPIED.name] == 1
    assert second_summary[DownloadStatus.IGNORED.name] == 2
    assert requests_mock.call_count == 1


def test_sha1_mismatch_is_reported(tmp_path: Path, requests_mock) -> None:
    url_json = tmp_path / "urls.json"
    out_root = tmp_path / "out"
    _write_urls(
        url_json,
        {
            "seq0": {
                "metadata_json": {
                    "filename": "NymeriaPlus_v1.0_seq0_metadata.json",
                    "sha1sum": "0" * 40,
                    "file_size_bytes": 3,
                    "download_url": "https://example.com/metadata.json",
                },
            },
        },
    )
    requests_mock.get("https://example.com/metadata.json", content=b"bad")
    license_source = tmp_path / "NYMERIAPLUS_DATASET_LICENSE"
    license_source.write_text("dataset license\n")

    manager = DownloadManager(url_json, out_root, license_source=license_source)
    summary = manager.download()

    assert summary[DownloadStatus.ERR_SHA1SUM.name] == 1
    assert not (out_root / "seq0/metadata.json").exists()
