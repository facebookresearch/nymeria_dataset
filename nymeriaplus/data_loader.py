# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level multimodal data loader for one NymeriaPlus sequence."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from nymeriaplus.layout import BodyModel, Recording, SequenceLayout
from nymeriaplus.loaders import (
    BoxyBBLoader,
    MHRBodyLoader,
    NarrationLoader,
    RecordingLoader,
    ShaperMeshLoader,
    SMPLBodyLoader,
    XSensBodyLoader,
)
from nymeriaplus.synchronized import SynchronizedSequence

logger = logging.getLogger(__name__)

# Sync warning thresholds (nanoseconds).
SLAM_POSE_TDIFF_THRESHOLD_NS: int = 3_000_000  # 3ms
SMPL_TDIFF_THRESHOLD_NS: int = 8_000_000  # 8ms (~2 frames @ 240fps)
MHR_TDIFF_THRESHOLD_NS: int = 8_000_000  # 8ms (~2 frames @ 240fps)
DEFAULT_TIMELINE_TRIM_NS: int = 1_000_000_000  # 1s shave at each end of overlap
XSENS_HEAD_IDX: int = 6


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - z * w)
    rot[..., 0, 2] = 2 * (x * z + y * w)
    rot[..., 1, 0] = 2 * (x * y + z * w)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - x * w)
    rot[..., 2, 0] = 2 * (x * z - y * w)
    rot[..., 2, 1] = 2 * (y * z + x * w)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _so3_log(rot: np.ndarray) -> np.ndarray:
    cos_t = (np.trace(rot) - 1.0) * 0.5
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if abs(theta) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(theta - np.pi) < 1e-6:
        mat = (rot + np.eye(3)) * 0.5
        idx = int(np.argmax(np.diag(mat)))
        axis = mat[:, idx] / np.sqrt(max(mat[idx, idx], 1e-12))
        return axis * theta
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ]
    ) / (2.0 * np.sin(theta))
    return axis * theta


def _solve_handeye_so3xR3(
    T_Wa_A: np.ndarray, T_Wb_B: np.ndarray, stride: int = 2
) -> np.ndarray:
    assert T_Wa_A.shape == T_Wb_B.shape
    n = len(T_Wa_A) - stride

    inv_Wa_A = np.linalg.inv(T_Wa_A[:n])
    inv_Wb_B = np.linalg.inv(T_Wb_B[:n])
    se3_A1_A2 = inv_Wa_A @ T_Wa_A[stride : stride + n]
    se3_B1_B2 = inv_Wb_B @ T_Wb_B[stride : stride + n]

    rot_A = se3_A1_A2[:, :3, :3]
    rot_B = se3_B1_B2[:, :3, :3]
    trans_A = se3_A1_A2[:, :3, 3]
    trans_B = se3_B1_B2[:, :3, 3]

    log_A = np.stack([_so3_log(rot_A[i]) for i in range(n)], axis=-1)
    log_B = np.stack([_so3_log(rot_B[i]) for i in range(n)], axis=-1)

    u, _s, vh = np.linalg.svd(log_B @ log_A.T, full_matrices=True)
    rot_X = vh.T @ u.T
    if np.linalg.det(rot_X) < 0:
        rot_X[2, :] *= -1.0

    eye3 = np.eye(3)
    jac = (rot_A - eye3).reshape(n * 3, 3)
    res = (rot_X @ trans_B[..., None]).squeeze(-1) - trans_A
    res = res.reshape(n * 3, 1)
    trans_X, *_ = np.linalg.lstsq(jac.T @ jac, jac.T @ res, rcond=None)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_X
    transform[:3, 3] = trans_X.flatten()
    return transform


class NymeriaPlusDataLoader:
    """Holds one loader per modality for a single sequence directory.

    Loaders that fail to find their file print a warning and store ``None``
    rather than raising, so partial sequences degrade gracefully.

    Call :meth:`synchronize_data` once you want a dense
    :class:`SynchronizedSequence` aligned to a fixed-rate timeline.
    """

    def __init__(
        self,
        sequence_root: Path,
        *,
        load_recordings: bool = True,
        load_smpl: bool = True,
        load_mhr: bool = False,
        load_xsens: bool = False,
        load_narration: bool = False,
        load_bbox: bool = False,
        load_mesh: bool = False,
        smpl_model_path: Path | None = None,
    ) -> None:
        if not sequence_root.is_dir():
            raise FileNotFoundError(f"sequence root not found: {sequence_root}")
        self.sequence_root = sequence_root
        self.seq_name = sequence_root.name

        # ---- Recordings (head, observer, lwrist, rwrist) -------------------
        self.recordings: dict[str, RecordingLoader] = {}
        if load_recordings:
            for rec in Recording:
                loader = RecordingLoader(sequence_root / rec.value)
                if loader.is_valid:
                    self.recordings[rec.value] = loader
                else:
                    logger.warning(f"recording invalid, skipping: {rec.value}")

        # ---- Body modalities ----------------------------------------------
        self.smpl_body: SMPLBodyLoader | None = None
        if load_smpl:
            loader = SMPLBodyLoader(
                sequence_root / SequenceLayout.BODY_XDATA_SMPL_NPZ,
                model_path=smpl_model_path,
            )
            self.smpl_body = loader if loader.is_valid else None

        self.mhr_body: MHRBodyLoader | None = None
        if load_mhr:
            loader = MHRBodyLoader(sequence_root / SequenceLayout.BODY_XDATA_MHR_GLB)
            self.mhr_body = loader if loader.is_valid else None

        self.xsens_body: XSensBodyLoader | None = None
        if load_xsens:
            loader = XSensBodyLoader(sequence_root / SequenceLayout.BODY_XDATA_NPZ)
            self.xsens_body = loader if loader.is_valid else None

        # ---- Other modalities ---------------------------------------------
        self.narration: NarrationLoader | None = None
        if load_narration:
            loader = NarrationLoader(sequence_root / "narration")
            self.narration = loader if loader.is_valid else None

        self.bbox: BoxyBBLoader | None = None
        if load_bbox:
            loader = BoxyBBLoader(sequence_root / SequenceLayout.OBJECTS_BOXY_DIR)
            self.bbox = loader if loader.is_valid else None

        self.mesh: ShaperMeshLoader | None = None
        if load_mesh:
            loader = ShaperMeshLoader(
                sequence_root / SequenceLayout.OBJECTS_SHAPER_DIR,
                instances_path=sequence_root
                / SequenceLayout.OBJECTS_BOXY_INSTANCES_JSON,
                category_palette=(
                    self.bbox.category_palette if self.bbox is not None else None
                ),
            )
            self.mesh = loader if loader.is_valid else None

    # -----------------------------------------------------------------------
    # Synchronization
    # -----------------------------------------------------------------------

    def synchronize_data(
        self,
        *,
        target_fps: float = 30.0,
        body_source: BodyModel | None = None,
        trim_ns: int = DEFAULT_TIMELINE_TRIM_NS,
    ) -> SynchronizedSequence:
        """Resample recordings + available body modalities to ``target_fps``.

        Mirrors the internal ``synchronize_sequence`` design: pick the
        overlap window across all VRS-bearing recordings and body
        timeline, shave ``trim_ns`` off each end, sample at ``target_fps``,
        return a dense :class:`SynchronizedSequence`.

        If ``body_source`` is ``None``, every loaded body modality is baked.
        Pass a specific ``BodyModel`` to bake only that body modality.
        """
        bodies = self._select_bodies(body_source)
        t0, t1 = self._compute_overlap_ns([body for _kind, body in bodies], trim_ns)

        dt_ns = int(1e9 / target_fps)
        sample_times_ns = np.arange(t0, t1, dt_ns, dtype=np.int64)
        n_frames = len(sample_times_ns)
        logger.info(
            f"[{self.seq_name}] overlap [{t0}, {t1}] "
            f"({(t1 - t0) / 1e9:.1f}s) -> {n_frames} frames @ {target_fps}fps"
        )

        synced = SynchronizedSequence(
            seq_name=self.seq_name, fps=target_fps, timestamps_ns=sample_times_ns
        )
        for kind, body in bodies:
            self._bake_body(synced, body, sample_times_ns, kind)
        self._bake_trajectories(synced, sample_times_ns)
        self._bake_xsens(synced, sample_times_ns)
        synced.report_quality()
        return synced

    # -----------------------------------------------------------------------
    # Sync helpers
    # -----------------------------------------------------------------------

    def _select_bodies(
        self, body_source: BodyModel | None
    ) -> list[tuple[BodyModel, SMPLBodyLoader | MHRBodyLoader]]:
        if body_source is None:
            bodies: list[tuple[BodyModel, SMPLBodyLoader | MHRBodyLoader]] = []
            if self.smpl_body is not None:
                bodies.append((BodyModel.SMPL, self.smpl_body))
            if self.mhr_body is not None:
                bodies.append((BodyModel.MHR, self.mhr_body))
            return bodies
        if body_source == BodyModel.SMPL:
            if self.smpl_body is None:
                logger.warning("body_source=SMPL but smpl_body is None")
                return []
            return [(BodyModel.SMPL, self.smpl_body)]
        if body_source == BodyModel.MHR:
            if self.mhr_body is None:
                logger.warning("body_source=MHR but mhr_body is None")
                return []
            return [(BodyModel.MHR, self.mhr_body)]
        raise ValueError(f"unknown body_source: {body_source}")

    def _compute_overlap_ns(
        self, bodies: list[SMPLBodyLoader | MHRBodyLoader], trim_ns: int
    ) -> tuple[int, int]:
        t_start: int | None = None
        t_end: int | None = None

        for body in bodies:
            if body.timespan_ns is None:
                continue
            bs, be = body.timespan_ns
            t_start = bs if t_start is None else max(t_start, bs)
            t_end = be if t_end is None else min(t_end, be)

        for rec in self.recordings.values():
            if not rec.has_vrs:
                continue
            rs, re = rec.get_global_timespan_ns()
            t_start = rs if t_start is None else max(t_start, rs)
            t_end = re if t_end is None else min(t_end, re)

        if t_start is None or t_end is None:
            raise RuntimeError(
                "nothing to synchronize: no body data and no VRS recordings"
            )

        t_start += trim_ns
        t_end -= trim_ns
        if t_start >= t_end:
            raise RuntimeError(f"empty overlap after trim: {t_start=} {t_end=}")
        return t_start, t_end

    def _bake_body(
        self,
        synced: SynchronizedSequence,
        body: SMPLBodyLoader | MHRBodyLoader | None,
        sample_times_ns: np.ndarray,
        body_source: BodyModel | None,
    ) -> None:
        if body is None:
            return

        idx = body.find_nearest_frames(sample_times_ns)
        if body_source == BodyModel.SMPL:
            assert isinstance(body, SMPLBodyLoader)
            synced.smpl_betas = body.betas[idx]
            synced.smpl_body_pose = body.body_pose[idx]
            synced.smpl_global_orient = body.global_orient[idx]
            synced.smpl_transl = body.transl[idx]
            synced.smpl_tdiff_ns = body.timestamps_ns[idx] - sample_times_ns

            n_warn = int(np.sum(np.abs(synced.smpl_tdiff_ns) > SMPL_TDIFF_THRESHOLD_NS))
            if n_warn:
                logger.warning(
                    f"  SMPL: {n_warn}/{len(sample_times_ns)} samples drift "
                    f"> {SMPL_TDIFF_THRESHOLD_NS / 1e6:.0f}ms"
                )
            return

        if body_source == BodyModel.MHR:
            assert isinstance(body, MHRBodyLoader)
            synced.mhr_frame_indices = idx.astype(np.int32)
            synced.mhr_tdiff_ns = body.timestamps_ns[idx] - sample_times_ns
            n_warn = int(np.sum(np.abs(synced.mhr_tdiff_ns) > MHR_TDIFF_THRESHOLD_NS))
            if n_warn:
                logger.warning(
                    f"  MHR: {n_warn}/{len(sample_times_ns)} samples drift "
                    f"> {MHR_TDIFF_THRESHOLD_NS / 1e6:.0f}ms"
                )

    def _bake_trajectories(
        self, synced: SynchronizedSequence, sample_times_ns: np.ndarray
    ) -> None:
        tag_to_field = {
            Recording.HEAD.value: ("T_world_head", "head_tdiff_ns"),
            Recording.LEFT_WRIST.value: ("T_world_lwrist", "lwrist_tdiff_ns"),
            Recording.RIGHT_WRIST.value: ("T_world_rwrist", "rwrist_tdiff_ns"),
            Recording.OBSERVER.value: ("T_world_observer", "observer_tdiff_ns"),
        }
        for tag, rec in self.recordings.items():
            if not (rec.has_vrs and rec.has_pose):
                continue
            traj_field, tdiff_field = tag_to_field[tag]
            traj, tdiff = rec.sample_trajectory_at_timecodes(sample_times_ns)
            setattr(synced, traj_field, traj)
            setattr(synced, tdiff_field, tdiff)
            n_warn = int(np.sum(np.abs(tdiff) > SLAM_POSE_TDIFF_THRESHOLD_NS))
            if n_warn:
                logger.warning(
                    f"  {tag}: {n_warn}/{len(sample_times_ns)} samples drift "
                    f"> {SLAM_POSE_TDIFF_THRESHOLD_NS / 1e6:.0f}ms"
                )

    def _bake_xsens(
        self, synced: SynchronizedSequence, sample_times_ns: np.ndarray
    ) -> None:
        body = self.xsens_body
        if (
            body is None
            or body.segment_positions is None
            or body.segment_quaternions_wxyz is None
            or body.timestamps_ns is None
        ):
            return
        idx = body.find_nearest_frames(sample_times_ns)
        pos_xsens = body.segment_positions[idx]
        quat_xsens = body.segment_quaternions_wxyz[idx]
        synced.xsens_tdiff_ns = body.timestamps_ns[idx] - sample_times_ns

        T_world_head = synced.T_world_head
        if T_world_head is None:
            synced.xsens_segment_positions = pos_xsens.astype(np.float32)
            logger.info("XSens kept in raw frame; no head trajectory for alignment")
            return

        rot_xsens_head = _quat_wxyz_to_rot(
            quat_xsens[:, XSENS_HEAD_IDX, :].astype(np.float64)
        )
        trans_xsens_head = pos_xsens[:, XSENS_HEAD_IDX, :].astype(np.float64)
        T_xsens_world_head = np.zeros((len(rot_xsens_head), 4, 4), dtype=np.float64)
        T_xsens_world_head[:, :3, :3] = rot_xsens_head
        T_xsens_world_head[:, :3, 3] = trans_xsens_head
        T_xsens_world_head[:, 3, 3] = 1.0

        T_aria_world_head = T_world_head.astype(np.float64)
        step = max(1, len(T_aria_world_head) // 5000)
        T_head_aria_head_xsens = _solve_handeye_so3xR3(
            T_aria_world_head[::step], T_xsens_world_head[::step], stride=2
        )

        T_aria_world_xsens_world = (
            T_aria_world_head
            @ T_head_aria_head_xsens
            @ np.linalg.inv(T_xsens_world_head)
        )
        rot = T_aria_world_xsens_world[:, :3, :3]
        trans = T_aria_world_xsens_world[:, :3, 3:4]
        pos_t = pos_xsens.astype(np.float64).transpose(0, 2, 1)
        synced.xsens_segment_positions = (
            (rot @ pos_t + trans).transpose(0, 2, 1).astype(np.float32)
        )

        head_residual = np.linalg.norm(
            synced.xsens_segment_positions[:, XSENS_HEAD_IDX]
            - T_aria_world_head[:, :3, 3],
            axis=1,
        ).mean()
        logger.info(
            "XSens aligned to Aria world via head hand-eye "
            f"(frames={len(T_aria_world_head[::step])}, residual={head_residual:.3f}m)"
        )
