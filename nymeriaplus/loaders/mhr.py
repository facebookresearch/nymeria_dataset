# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for MHR (Momentum Human Rig) body meshes stored in GLB files."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MHRBodyLoader:
    """Loads ``body/xdata_mhr.glb`` and skins per-frame mesh vertices.

    MHR GLB motion is stored in Momentum convention (Y-up, centimeters).
    Returned vertices are converted to the viewer/world convention (Z-up,
    meters), matching the original Nymeria OSS MHR provider.
    """

    _MOMENTUM_TO_VIEWER = np.array(
        [[0.01, 0.0, 0.0], [0.0, 0.0, -0.01], [0.0, 0.01, 0.0]],
        dtype=np.float64,
    )

    def __init__(self, glb_path: Path) -> None:
        self.glb_path = glb_path
        self.character: Any = None
        self.motion: np.ndarray | None = None
        self.identity: np.ndarray | None = None
        self.fps: float | None = None
        self.timestamps_ns: np.ndarray | None = None
        self._pymomentum: Any = None

        if not glb_path.is_file():
            logger.warning(f"MHR GLB missing: {glb_path}")
            return

        try:
            # Load torch first so its bundled libtorch symbols are available
            # before pymomentum's compiled geometry extension initializes.
            importlib.import_module("torch")
            import pymomentum as pym
            import pymomentum.geometry as pym_geometry
        except ImportError as e:
            logger.warning(f"pymomentum is required for MHR loading: {e}")
            return

        try:
            character, motion, identity, fps = (
                pym_geometry.Character.load_gltf_with_motion(str(glb_path))
            )
            timestamps_us = pym_geometry.Character.load_motion_timestamps(str(glb_path))
        except Exception as e:
            logger.warning(f"failed to load MHR GLB {glb_path}: {e}")
            return

        if not character.has_mesh:
            logger.warning(f"MHR GLB has no mesh: {glb_path}")
            return

        self._pymomentum = pym
        self.character = character
        self.motion = np.asarray(motion, dtype=np.float32)
        self.identity = None if identity is None else np.asarray(identity)
        self.fps = float(fps)
        if timestamps_us:
            self.timestamps_ns = np.asarray(timestamps_us, dtype=np.int64) * 1000
        else:
            dt_ns = int(round(1e9 / self.fps))
            self.timestamps_ns = np.arange(self.motion.shape[0], dtype=np.int64) * dt_ns
            logger.warning(
                f"MHR GLB has no motion timestamps; using synthetic {self.fps:g}fps "
                "timestamps starting at zero"
            )

        logger.info(
            f"loaded MHR from {glb_path}: frames={self.num_frames}, "
            f"fps={self.fps:g}, motion_shape={self.motion.shape}"
        )

    @property
    def is_valid(self) -> bool:
        return self.character is not None and self.motion is not None

    @property
    def num_frames(self) -> int:
        return 0 if self.motion is None else int(self.motion.shape[0])

    @property
    def timespan_ns(self) -> tuple[int, int] | None:
        if self.timestamps_ns is None:
            return None
        return int(self.timestamps_ns[0]), int(self.timestamps_ns[-1])

    @property
    def faces(self) -> np.ndarray:
        if self.character is None:
            raise RuntimeError("MHR data not loaded")
        faces = np.asarray(self.character.mesh.faces, dtype=np.uint32)
        if faces.ndim == 1:
            faces = faces.reshape(-1, 3)
        return faces

    @property
    def num_vertices(self) -> int:
        if self.character is None:
            return 0
        vertices = getattr(self.character.mesh, "vertices", None)
        if vertices is not None:
            return int(np.asarray(vertices).reshape(-1, 3).shape[0])
        faces = self.faces
        return int(faces.max()) + 1 if faces.size else 0

    @property
    def skeleton_parents(self) -> np.ndarray:
        if self.character is None:
            return np.zeros(0, dtype=np.int32)
        return np.asarray(self.character.skeleton.joint_parents, dtype=np.int32)

    def find_nearest_frames(self, query_ns: np.ndarray) -> np.ndarray:
        """Vectorised nearest-frame lookup over GLB motion timestamps."""
        if self.timestamps_ns is None:
            raise RuntimeError("MHR data not loaded")
        idx = np.searchsorted(self.timestamps_ns, query_ns)
        idx = np.clip(idx, 0, self.num_frames - 1)
        left = np.clip(idx - 1, 0, self.num_frames - 1)
        dist_right = np.abs(self.timestamps_ns[idx] - query_ns)
        dist_left = np.abs(self.timestamps_ns[left] - query_ns)
        use_left = dist_left <= dist_right
        idx[use_left] = left[use_left]
        return idx

    def _skeleton_state(self, frame_idx: int) -> np.ndarray:
        if self.character is None or self.motion is None or self._pymomentum is None:
            raise RuntimeError("MHR data not loaded")
        motion = self.motion[int(frame_idx)]
        return self._pymomentum.geometry.model_parameters_to_skeleton_state(
            self.character, motion
        )

    def _to_viewer_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return (self._MOMENTUM_TO_VIEWER @ points.T).T.astype(np.float32)

    def forward(self, frame_idx: int) -> np.ndarray:
        """Return posed MHR mesh vertices ``(V, 3)`` in Z-up meters."""
        if self.character is None:
            raise RuntimeError("MHR data not loaded")
        skel_state = self._skeleton_state(frame_idx)
        skin = self.character.skin_points(skel_state)
        return self._to_viewer_points(skin)

    def skeleton_joints(self, frame_idx: int) -> np.ndarray:
        """Return posed MHR skeleton joint positions ``(J, 3)`` in Z-up meters."""
        skel_state = self._skeleton_state(frame_idx)
        return self._to_viewer_points(skel_state[:, :3])

    def forward_with_joints(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return posed MHR mesh vertices and skeleton joints in Z-up meters."""
        if self.character is None:
            raise RuntimeError("MHR data not loaded")
        skel_state = self._skeleton_state(frame_idx)
        skin = self.character.skin_points(skel_state)
        return self._to_viewer_points(skin), self._to_viewer_points(skel_state[:, :3])
