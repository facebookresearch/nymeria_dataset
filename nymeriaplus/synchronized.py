# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dense, time-aligned multi-modal data for one NymeriaPlus sequence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SynchronizedSequence:
    """All modalities aligned along axis 0 to a single timestamp array.

    Pose arrays are 4x4 SE(3) matrices in Aria-world (Z-up, meters).
    Save with :meth:`save_npz` / load with :meth:`load_npz`.
    """

    seq_name: str
    fps: float
    timestamps_ns: np.ndarray  # (N,) int64

    # SMPL block (None if SMPL absent)
    smpl_betas: np.ndarray | None = None  # (N, 10)
    smpl_body_pose: np.ndarray | None = None  # (N, 69)
    smpl_global_orient: np.ndarray | None = None  # (N, 3)
    smpl_transl: np.ndarray | None = None  # (N, 3)
    smpl_tdiff_ns: np.ndarray | None = None  # (N,) drift to nearest source sample

    # MHR block (None if MHR absent)
    mhr_frame_indices: np.ndarray | None = None  # (N,) nearest GLB motion frames
    mhr_tdiff_ns: np.ndarray | None = None  # (N,) drift to nearest source sample

    # XSens block (None if XSens absent)
    xsens_segment_positions: np.ndarray | None = None  # (N, 23, 3)
    xsens_tdiff_ns: np.ndarray | None = None  # (N,) drift to nearest source sample

    # Per-recording trajectories (None if recording absent or missing pose)
    T_world_head: np.ndarray | None = None  # (N, 4, 4)
    T_world_lwrist: np.ndarray | None = None
    T_world_rwrist: np.ndarray | None = None
    T_world_observer: np.ndarray | None = None
    head_tdiff_ns: np.ndarray | None = None
    lwrist_tdiff_ns: np.ndarray | None = None
    rwrist_tdiff_ns: np.ndarray | None = None
    observer_tdiff_ns: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def save_npz(self, path: Path | str) -> None:
        arrays: dict[str, np.ndarray] = {
            "fps": np.array([self.fps], dtype=np.float32),
            "seq_name": np.array([self.seq_name]),
            "timestamps_ns": self.timestamps_ns,
        }
        for name in (
            "smpl_betas",
            "smpl_body_pose",
            "smpl_global_orient",
            "smpl_transl",
            "smpl_tdiff_ns",
            "mhr_frame_indices",
            "mhr_tdiff_ns",
            "xsens_segment_positions",
            "xsens_tdiff_ns",
            "T_world_head",
            "T_world_lwrist",
            "T_world_rwrist",
            "T_world_observer",
            "head_tdiff_ns",
            "lwrist_tdiff_ns",
            "rwrist_tdiff_ns",
            "observer_tdiff_ns",
        ):
            v = getattr(self, name)
            if v is not None:
                arrays[name] = v
        np.savez(path, **arrays)
        logger.info(f"saved synchronized sequence to {path}")

    @classmethod
    def load_npz(cls, path: Path | str) -> SynchronizedSequence:
        data = dict(np.load(path, allow_pickle=False))
        kwargs: dict[str, Any] = {
            "seq_name": str(data["seq_name"][0]),
            "fps": float(data["fps"][0]),
            "timestamps_ns": data["timestamps_ns"],
        }
        for k, v in data.items():
            if k in ("seq_name", "fps", "timestamps_ns"):
                continue
            kwargs[k] = v
        return cls(**kwargs)

    def report_quality(self) -> None:
        for name, arr in (
            ("smpl", self.smpl_tdiff_ns),
            ("mhr", self.mhr_tdiff_ns),
            ("xsens", self.xsens_tdiff_ns),
            ("head", self.head_tdiff_ns),
            ("lwrist", self.lwrist_tdiff_ns),
            ("rwrist", self.rwrist_tdiff_ns),
            ("observer", self.observer_tdiff_ns),
        ):
            if arr is None:
                continue
            ms = np.abs(arr) / 1e6
            logger.info(
                f"  {name:8s} tdiff: mean={ms.mean():.3f}ms "
                f"max={ms.max():.3f}ms p99={np.percentile(ms, 99):.3f}ms"
            )
