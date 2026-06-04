# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for SMPL body parameters (per-frame)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SMPLBodyLoader:
    """Loads ``body/xdata_smpl_neutral.npz`` (already in Aria-world frame).

    Never raises in ``__init__`` -- inspect :attr:`is_valid` after construction.
    The ``smplx`` model is loaded lazily on first :meth:`forward` call so a
    user that only needs raw parameters doesn't pay for it.

    Expected npz keys (one row per frame):
        betas         (num_frames, 10)
        body_pose     (num_frames, 69)
        global_orient (num_frames, 3)
        transl        (num_frames, 3)
        timestamps    (num_frames,)   in microseconds
    """

    def __init__(self, npz_path: Path, model_path: Path | None = None) -> None:
        self.npz_path = npz_path
        self.model_path = model_path

        self.timestamps_ns: np.ndarray | None = None
        self.betas: np.ndarray | None = None
        self.body_pose: np.ndarray | None = None
        self.global_orient: np.ndarray | None = None
        self.transl: np.ndarray | None = None

        if not npz_path.is_file():
            logger.warning(f"SMPL npz missing: {npz_path}")
            return

        logger.info(f"loading SMPL params from {npz_path}")
        data = dict(np.load(npz_path))
        self.timestamps_ns = (data["timestamps"] * 1000).astype(np.int64)
        self.betas = data["betas"].astype(np.float32)
        self.body_pose = data["body_pose"].astype(np.float32)
        self.global_orient = data["global_orient"].astype(np.float32)
        self.transl = data["transl"].astype(np.float32)

        self._model: Any = None  # smplx.SMPL once loaded

    # ------------------------------------------------------------------ flags

    @property
    def is_valid(self) -> bool:
        return self.timestamps_ns is not None

    @property
    def num_frames(self) -> int:
        return 0 if self.timestamps_ns is None else int(self.timestamps_ns.shape[0])

    @property
    def timespan_ns(self) -> tuple[int, int] | None:
        if self.timestamps_ns is None:
            return None
        return int(self.timestamps_ns[0]), int(self.timestamps_ns[-1])

    # ----------------------------------------------------------------- query

    def find_nearest_frames(self, query_ns: np.ndarray) -> np.ndarray:
        """Vectorised nearest-frame lookup over the actual stored timestamps."""
        if self.timestamps_ns is None:
            raise RuntimeError("SMPL data not loaded")
        idx = np.searchsorted(self.timestamps_ns, query_ns)
        idx = np.clip(idx, 0, self.num_frames - 1)
        left = np.clip(idx - 1, 0, self.num_frames - 1)
        dist_right = np.abs(self.timestamps_ns[idx] - query_ns)
        dist_left = np.abs(self.timestamps_ns[left] - query_ns)
        use_left = dist_left <= dist_right
        idx[use_left] = left[use_left]
        return idx

    # --------------------------------------------------------------- posing

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None or not self.model_path.is_file():
            raise FileNotFoundError(
                f"SMPL .pkl model not found: {self.model_path}; "
                "pass model_path to the loader to enable posing"
            )
        try:
            import smplx
        except ImportError as e:
            raise ImportError(
                "smplx is required for SMPL posing; install nymeriaplus[smpl]"
            ) from e
        logger.info(f"loading SMPL model from {self.model_path}")
        self._model = smplx.create(
            str(self.model_path), model_type="smpl", batch_size=1
        )

    @property
    def faces(self) -> np.ndarray:
        self._ensure_model()
        return self._model.faces

    def forward(self, frame_idx: int) -> np.ndarray:
        """Return posed SMPL vertices ``(V, 3)`` for one frame, in the
        canonical Aria-world frame (no extra correction applied).
        """
        if self.body_pose is None:
            raise RuntimeError("SMPL data not loaded")
        self._ensure_model()
        import torch

        sl = slice(frame_idx, frame_idx + 1)
        with torch.no_grad():
            out = self._model(
                betas=torch.as_tensor(self.betas[sl], dtype=torch.float32),
                body_pose=torch.as_tensor(self.body_pose[sl], dtype=torch.float32),
                global_orient=torch.as_tensor(
                    self.global_orient[sl], dtype=torch.float32
                ),
                transl=torch.as_tensor(self.transl[sl], dtype=torch.float32),
            )
        return out.vertices.squeeze(0).cpu().numpy()
