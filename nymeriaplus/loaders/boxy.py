# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for Boxy 3D bounding boxes."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nymeriaplus.loaders.object_palette import (
    build_category_palette,
    CategoryPaletteEntry,
    generate_category_color,
)

logger = logging.getLogger(__name__)

_EDGE_INDICES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
_BBOX_VISIBILITY_THRESHOLD = 0.0001


@dataclass(frozen=True)
class Boxy2DBB:
    object_uid: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    visibility: float


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def nearest_within_tol(
    timestamps: np.ndarray | None, query: int, tol_ns: int
) -> int | None:
    """Nearest value in sorted ``timestamps`` to ``query``, or ``None`` if the
    array is empty or the closest entry is farther than ``tol_ns``."""
    if timestamps is None or timestamps.size == 0:
        return None
    idx = int(np.searchsorted(timestamps, query))
    candidates = []
    if idx < len(timestamps):
        candidates.append(int(timestamps[idx]))
    if idx > 0:
        candidates.append(int(timestamps[idx - 1]))
    if not candidates:
        return None
    nearest = min(candidates, key=lambda ts: abs(ts - query))
    if abs(nearest - query) > tol_ns:
        return None
    return nearest


def _read_csv_by_object_uid(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp, skipinitialspace=True)
        return {int(row["object_uid"]): row for row in reader}


class BoxyBBLoader:
    """Load static 3D bounding boxes from ``objects/boxy`` release files."""

    def __init__(self, boxy_dir: Path) -> None:
        self.boxy_dir = boxy_dir
        self.category_palette: dict[str, CategoryPaletteEntry] = {}
        self.object_labels: dict[int, str] = {}
        self.object_colors: dict[int, tuple[float, float, float]] = {}
        self.labels: list[str] = []
        self.object_uids: list[int] = []
        self.edges = np.zeros((0, 12, 2, 3), dtype=np.float32)
        self.centers = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.line_points = np.zeros((0, 3), dtype=np.float32)
        self.line_colors = np.zeros((0, 3), dtype=np.float32)
        self.bb2d_by_recording: dict[str, dict[int, list[Boxy2DBB]]] = {}
        self.bb2d_timestamps: dict[str, np.ndarray] = {}
        # Per-stream object visibility for the non-RGB cameras (SLAM), used for
        # camera-selectable "visible only" 3D filtering. RGB (214-1) visibility
        # is derived on demand from ``bb2d_by_recording`` (see ``_rgb_visibility``)
        # to avoid storing it twice. tag -> stream_id -> {timestamp: {uid}}.
        self.visibility_by_recording: dict[str, dict[str, dict[int, set[int]]]] = {}
        # tag -> stream_id -> sorted timestamp array (device time, ns).
        self.visibility_timestamps: dict[str, dict[str, np.ndarray]] = {}
        # tag -> (sorted timestamps, {timestamp: {uid}}) cache for RGB.
        self._rgb_visibility_cache: dict[
            str, tuple[np.ndarray, dict[int, set[int]]]
        ] = {}
        self._is_valid = False

        bbox_csv = boxy_dir / "3dbb.csv"
        objects_csv = boxy_dir / "scene_objects.csv"
        instances_json = boxy_dir / "instances.json"
        required = (bbox_csv, objects_csv, instances_json)
        if not boxy_dir.is_dir() or not all(path.is_file() for path in required):
            logger.warning(f"Boxy 3D bounding boxes unavailable in {boxy_dir}")
            return

        with instances_json.open() as fp:
            self.instances: dict[str, dict] = json.load(fp)
        self.category_palette = build_category_palette(self.instances)

        bbox_rows = _read_csv_by_object_uid(bbox_csv)
        object_rows = _read_csv_by_object_uid(objects_csv)
        self.object_uids = sorted(set(bbox_rows) & set(object_rows))
        if not self.object_uids:
            logger.warning(
                f"No Boxy objects have both bbox and pose data in {boxy_dir}"
            )
            return

        edges: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        labels: list[str] = []
        for object_uid in self.object_uids:
            edges.append(
                self._build_edges(bbox_rows[object_uid], object_rows[object_uid])
            )
            instance = self.instances.get(str(object_uid), {})
            category = str(instance.get("category", "Unknown"))
            instance_id = int(instance.get("instance_id", object_uid))
            label = f"{category}_{instance_id}"
            labels.append(label)
            category_uid = int(instance.get("category_uid", 0))
            color = tuple(
                float(v)
                for v in generate_category_color(category_uid).astype(np.float32)
                / 255.0
            )
            colors.append(np.array(color, dtype=np.float32))
            self.object_labels[object_uid] = label
            self.object_colors[object_uid] = color

        self.edges = np.stack(edges, axis=0)
        self.centers = self.edges.reshape(len(self.object_uids), -1, 3).mean(axis=1)
        self.colors = np.stack(colors, axis=0)
        self.labels = labels
        self.line_points = self.edges.reshape(-1, 3).astype(np.float32)
        self.line_colors = np.repeat(self.colors[:, None, :], 24, axis=1).reshape(-1, 3)
        self._load_2d_bounding_boxes()
        self._is_valid = True
        logger.info(f"Loaded {len(self.object_uids)} Boxy 3D bounding boxes")

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    def _build_edges(
        self, bbox_row: dict[str, str], object_row: dict[str, str]
    ) -> np.ndarray:
        xmin = float(bbox_row["p_local_obj_xmin[m]"])
        xmax = float(bbox_row["p_local_obj_xmax[m]"])
        ymin = float(bbox_row["p_local_obj_ymin[m]"])
        ymax = float(bbox_row["p_local_obj_ymax[m]"])
        zmin = float(bbox_row["p_local_obj_zmin[m]"])
        zmax = float(bbox_row["p_local_obj_zmax[m]"])
        corners_local = np.array(
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ],
            dtype=np.float32,
        )
        q = np.array(
            [
                float(object_row["q_wo_w"]),
                float(object_row["q_wo_x"]),
                float(object_row["q_wo_y"]),
                float(object_row["q_wo_z"]),
            ],
            dtype=np.float32,
        )
        rot = _quat_wxyz_to_rot(q)
        trans = np.array(
            [
                float(object_row["t_wo_x[m]"]),
                float(object_row["t_wo_y[m]"]),
                float(object_row["t_wo_z[m]"]),
            ],
            dtype=np.float32,
        )
        corners_world = corners_local @ rot.T + trans
        return np.array(
            [
                [corners_world[start], corners_world[end]]
                for start, end in _EDGE_INDICES
            ],
            dtype=np.float32,
        )

    def _load_2d_bounding_boxes(self) -> None:
        for tag, filename in (
            ("head", "2dbb_recording_head.csv"),
            ("observer", "2dbb_recording_observer.csv"),
        ):
            path = self.boxy_dir / filename
            if not path.is_file():
                continue
            by_timestamp, vis_by_stream = self._read_2d_bounding_box_csv(path)
            if by_timestamp:
                self.bb2d_by_recording[tag] = by_timestamp
                self.bb2d_timestamps[tag] = np.array(
                    sorted(by_timestamp.keys()), dtype=np.int64
                )
                logger.info(
                    f"Loaded {sum(len(v) for v in by_timestamp.values())} "
                    f"Boxy 2D boxes for {tag} RGB"
                )
            if vis_by_stream:
                self.visibility_by_recording[tag] = vis_by_stream
                self.visibility_timestamps[tag] = {
                    stream: np.array(sorted(per_ts.keys()), dtype=np.int64)
                    for stream, per_ts in vis_by_stream.items()
                }
                logger.info(
                    f"Loaded Boxy 2D visibility for {tag}: "
                    f"streams {sorted(vis_by_stream)}"
                )

    def _read_2d_bounding_box_csv(
        self, path: Path
    ) -> tuple[dict[int, list[Boxy2DBB]], dict[str, dict[int, set[int]]]]:
        """Parse one 2D-bbox CSV.

        Returns ``(rgb_by_timestamp, visibility_by_stream)`` where the first is
        the RGB-only (``214-1``) full boxes used by the projection overlay, and
        the second maps every non-RGB ``stream_id`` to
        ``{timestamp: {object_uid}}`` for camera-selectable visibility queries.
        RGB visibility is not duplicated here; it is derived from the first map.
        """
        by_timestamp: dict[int, list[Boxy2DBB]] = {}
        vis_by_stream: dict[str, dict[int, set[int]]] = {}
        with path.open(newline="") as fp:
            reader = csv.DictReader(fp, skipinitialspace=True)
            for row in reader:
                stream_id = row.get("stream_id")
                try:
                    object_uid = int(row["object_uid"])
                    timestamp = int(row["timestamp[ns]"])
                    xmin = max(0.0, float(row["x_min[pixel]"]))
                    xmax = max(0.0, float(row["x_max[pixel]"]))
                    ymin = max(0.0, float(row["y_min[pixel]"]))
                    ymax = max(0.0, float(row["y_max[pixel]"]))
                    visibility = float(row["visibility_ratio[%]"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(
                    np.isfinite(v) for v in (xmin, xmax, ymin, ymax, visibility)
                ):
                    continue
                if visibility < _BBOX_VISIBILITY_THRESHOLD:
                    continue
                if stream_id == "214-1":
                    by_timestamp.setdefault(timestamp, []).append(
                        Boxy2DBB(
                            object_uid=object_uid,
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                            visibility=visibility,
                        )
                    )
                elif stream_id:
                    vis_by_stream.setdefault(stream_id, {}).setdefault(
                        timestamp, set()
                    ).add(object_uid)
        return by_timestamp, vis_by_stream

    def _rgb_visibility(
        self, tag: str
    ) -> tuple[np.ndarray, dict[int, set[int]]] | None:
        """RGB (``214-1``) visibility as ``(sorted timestamps, {ts: {uid}})``,
        derived once from ``bb2d_by_recording`` and cached."""
        cached = self._rgb_visibility_cache.get(tag)
        if cached is not None:
            return cached
        by_timestamp = self.bb2d_by_recording.get(tag)
        timestamps = self.bb2d_timestamps.get(tag)
        if by_timestamp is None or timestamps is None:
            return None
        mapping = {
            ts: {box.object_uid for box in boxes} for ts, boxes in by_timestamp.items()
        }
        cached = (timestamps, mapping)
        self._rgb_visibility_cache[tag] = cached
        return cached

    def visibility_tags(self) -> set[str]:
        """Recording tags for which any camera visibility data is available."""
        return set(self.visibility_by_recording) | set(self.bb2d_by_recording)

    def _stream_visibility(
        self, tag: str, stream: str
    ) -> tuple[np.ndarray, dict[int, set[int]]] | None:
        if stream == "214-1":
            return self._rgb_visibility(tag)
        timestamps = self.visibility_timestamps.get(tag, {}).get(stream)
        mapping = self.visibility_by_recording.get(tag, {}).get(stream)
        if timestamps is None or mapping is None:
            return None
        return timestamps, mapping

    def visible_object_uids(
        self,
        tag: str,
        device_time_ns: int,
        streams: tuple[str, ...],
        tol_ns: int = 50_000_000,
    ) -> set[int]:
        """Union of object_uids visible in any of ``streams`` of recording
        ``tag`` at ``device_time_ns`` (nearest annotated frame within tol)."""
        visible: set[int] = set()
        for stream in streams:
            stream_vis = self._stream_visibility(tag, stream)
            if stream_vis is None:
                continue
            timestamps, mapping = stream_vis
            nearest_ts = nearest_within_tol(timestamps, device_time_ns, tol_ns)
            if nearest_ts is None:
                continue
            visible |= mapping.get(nearest_ts, set())
        return visible
