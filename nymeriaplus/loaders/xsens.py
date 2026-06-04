# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for raw XSens body capture."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class XSensBodyLoader:
    """Loads ``body/xdata.npz`` segment positions and orientations.

    Expected keys:
        segment_tXYZ  (num_frames, segment_count * 3)
        segment_qWXYZ (num_frames, segment_count * 4)
        timestamps_us (num_frames,)
    """

    _DT_NOMINAL_US: float = 1.0e6 / 240.0
    _DT_TOL_US: int = 1000
    _T_CORRECT_TOL_US: int = 10_000

    def __init__(self, npz_path: Path) -> None:
        self.npz_path = npz_path
        self.timestamps_ns: np.ndarray | None = None
        self.segment_positions: np.ndarray | None = None
        self.segment_quaternions_wxyz: np.ndarray | None = None

        if not npz_path.is_file():
            logger.warning(f"XSens npz missing: {npz_path}")
            return

        logger.info(f"loading XSens params from {npz_path}")
        data = dict(np.load(npz_path))
        self._correct_timestamps(data)
        self._correct_quaternions(data)
        self.timestamps_ns = (data["timestamps_us"] * 1000).astype(np.int64)
        segment_count = int(np.asarray(data.get("segmentCount", [23]))[0])
        self.segment_positions = (
            data["segment_tXYZ"].astype(np.float32).reshape(-1, segment_count, 3)
        )
        self.segment_quaternions_wxyz = (
            data["segment_qWXYZ"].astype(np.float32).reshape(-1, segment_count, 4)
        )

    def _correct_timestamps(self, data: dict[str, np.ndarray]) -> None:
        timestamps_us = data["timestamps_us"]
        dt = timestamps_us[1:] - timestamps_us[:-1]
        invalid = np.abs(dt - self._DT_NOMINAL_US) > self._DT_TOL_US
        if not int(invalid.sum()):
            return

        logger.warning(
            f"XSens: {int(invalid.sum())} invalid timestamps; patching cadence"
        )
        dt_corr = dt.copy()
        dt_corr[invalid] = int(self._DT_NOMINAL_US)
        dt_corr = np.insert(dt_corr, 0, 0)
        corrected = timestamps_us[0] + np.cumsum(dt_corr)
        max_drift = int(abs(corrected[-1] - timestamps_us[-1]))
        if max_drift > self._T_CORRECT_TOL_US:
            raise RuntimeError(
                f"XSens timestamp correction drift {max_drift}us exceeds tolerance"
            )
        data["timestamps_us"] = corrected

    def _correct_quaternions(self, data: dict[str, np.ndarray]) -> None:
        segment_count = int(np.asarray(data.get("segmentCount", [23]))[0])
        q = data["segment_qWXYZ"].reshape(-1, segment_count, 4)
        qn = np.linalg.norm(q, axis=-1)
        invalid = qn < 0.1
        if not int(invalid.sum()):
            return

        logger.warning(f"XSens: {int(invalid.sum())} invalid quaternions; patching")
        for part_idx in range(segment_count):
            if qn[0, part_idx] < 0.5:
                q[0, part_idx] = np.array([1, 0, 0, 0], dtype=q.dtype)
        for frame_idx in range(1, qn.shape[0]):
            for part_idx in range(segment_count):
                if qn[frame_idx, part_idx] < 0.5:
                    q[frame_idx, part_idx] = q[frame_idx - 1, part_idx]
        data["segment_qWXYZ"] = q.reshape(-1, segment_count * 4)

    @property
    def is_valid(self) -> bool:
        return self.timestamps_ns is not None and self.segment_positions is not None

    @property
    def num_frames(self) -> int:
        return 0 if self.timestamps_ns is None else int(self.timestamps_ns.shape[0])

    @property
    def timespan_ns(self) -> tuple[int, int] | None:
        if self.timestamps_ns is None:
            return None
        return int(self.timestamps_ns[0]), int(self.timestamps_ns[-1])

    def find_nearest_frames(self, query_ns: np.ndarray) -> np.ndarray:
        if self.timestamps_ns is None:
            raise RuntimeError("XSens data not loaded")
        idx = np.searchsorted(self.timestamps_ns, query_ns)
        idx = np.clip(idx, 0, self.num_frames - 1)
        left = np.clip(idx - 1, 0, self.num_frames - 1)
        dist_right = np.abs(self.timestamps_ns[idx] - query_ns)
        dist_left = np.abs(self.timestamps_ns[left] - query_ns)
        use_left = dist_left <= dist_right
        idx[use_left] = left[use_left]
        return idx
