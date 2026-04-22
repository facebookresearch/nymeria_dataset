# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
from loguru import logger

try:
    import smplx
    import torch

    _HAS_SMPLX = True
except ImportError:
    _HAS_SMPLX = False


class SmplDataProvider:
    """Data provider for SMPL body model parameters.

    Loads per-frame SMPL parameters from an .npz file and a SMPL model
    from a .pkl file.

    The npz file is expected to contain the following keys:
        betas        (num_frames, 10)
        body_pose    (num_frames, 69)
        global_orient(num_frames, 3)
        transl       (num_frames, 3)
        timestamps   (num_frames,)
    """

    def __init__(self, npzfile: str, model_path: str) -> None:
        if not _HAS_SMPLX:
            raise ImportError("Missing smplx and torch packages. ")

        if not Path(npzfile).is_file():
            raise FileNotFoundError(f"SMPL npz file not found: {npzfile}")

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"SMPL model file not found: {model_path}. ")

        logger.info(f"loading SMPL data from {npzfile}")
        data = dict(np.load(npzfile))
        for k, v in data.items():
            logger.info(f"{k}, shape={v.shape}")

        self.betas: np.ndarray = data["betas"]
        self.body_pose: np.ndarray = data["body_pose"]
        self.global_orient: np.ndarray = data["global_orient"]
        self.transl: np.ndarray = data["transl"]
        self.timestamps: np.ndarray = data["timestamps"]

        self.num_frames: int = self.body_pose.shape[0]

        logger.info(f"loading SMPL model from {model_path}")
        self._model = smplx.create(
            model_path,
            model_type="smpl",
            batch_size=1,
        )

    @property
    def faces(self) -> np.ndarray:
        """Triangle face indices from the SMPL model."""
        return self._model.faces

    def get_global_timespan_us(self) -> tuple[int, int]:
        """Return the first and last timestamp in microseconds."""
        return int(self.timestamps[0]), int(self.timestamps[-1])

    def get_posed_skin(self, frame_idx: int) -> np.ndarray:
        """Return posed SMPL vertices for a single frame.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            Vertices as a numpy array of shape (N, 3) in Aria world space
            (meters).
        """
        betas = torch.tensor(self.betas[frame_idx : frame_idx + 1], dtype=torch.float32)
        body_pose = torch.tensor(
            self.body_pose[frame_idx : frame_idx + 1], dtype=torch.float32
        )
        global_orient = torch.tensor(
            self.global_orient[frame_idx : frame_idx + 1], dtype=torch.float32
        )
        transl = torch.tensor(
            self.transl[frame_idx : frame_idx + 1], dtype=torch.float32
        )

        with torch.no_grad():
            output = self._model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )

        vertices: np.ndarray = output.vertices.squeeze(0).cpu().numpy()
        return vertices


def create_smpl_data_provider(
    npzfile: str, model_path: str
) -> "SmplDataProvider | None":
    """Factory function to create a SmplDataProvider.

    Returns None if the npz file does not exist, the smplx package is not
    installed, or creation fails for any other reason.
    """
    if not Path(npzfile).is_file():
        logger.warning(f"SMPL npz file not found: {npzfile}")
        return None

    if not Path(model_path).is_file():
        logger.warning(f"SMPL model file not found: {model_path}")
        return None

    try:
        return SmplDataProvider(npzfile=npzfile, model_path=model_path)
    except Exception as e:
        logger.error(f"Failed to create SmplDataProvider: {e}")
        return None
