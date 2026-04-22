# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pymomentum as pym
from loguru import logger
from pymomentum.geometry import Character, Mesh


class MhrDataProvider:
    """Data provider for MHR (Momentum Human Rig) body mesh stored in GLB files.

    MHR vertices are in Momentum convention (Y-up, centimeters).
    ``get_posed_skin`` converts them to XSens/viewer convention (Z-up, meters).
    Timestamps are already synchronized with Aria -- direct frame index access.
    """

    # Full Momentum→XSens transform: axis swap (Y-up → Z-up) + cm→m scaling.
    _A_Wx_Wm: np.ndarray = np.array(
        [[0.01, 0, 0], [0, 0, -0.01], [0, 0.01, 0]], dtype=np.float64
    )

    def __init__(self, glbfile: str) -> None:
        logger.info(f"loading MHR from {glbfile=}")
        self.character, self.motion, _, self.fps = Character.load_gltf_with_motion(
            glbfile
        )
        assert self.character.has_mesh
        self.num_frames: int = self.motion.shape[0]
        logger.info(
            f"MHR loaded: {self.num_frames} frames, {self.fps} fps, "
            f"motion shape {self.motion.shape}"
        )

    @property
    def template_mesh(self) -> Mesh:
        return self.character.mesh

    def get_posed_skin(self, frame_idx: int) -> np.ndarray:
        """Return posed mesh vertices for a single frame.

        Args:
            frame_idx: Frame index into the motion sequence.

        Returns:
            Vertex positions as an (N, 3) numpy array in XSens convention
            (Z-up, meters).
        """
        motion = self.motion[frame_idx]
        skel_state = pym.geometry.model_parameters_to_skeleton_state(
            self.character, motion
        )
        skin = self.character.skin_points(skel_state)
        vertices: np.ndarray = np.asarray(skin)
        return (self._A_Wx_Wm @ vertices.T).T


def create_mhr_data_provider(glbfile: str) -> MhrDataProvider | None:
    if Path(glbfile).is_file():
        return MhrDataProvider(glbfile=glbfile)
    else:
        logger.warning(f"MHR GLB file not found: {glbfile}")
        return None
