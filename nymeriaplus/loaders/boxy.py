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
            by_timestamp = self._read_2d_bounding_box_csv(path)
            if not by_timestamp:
                continue
            self.bb2d_by_recording[tag] = by_timestamp
            self.bb2d_timestamps[tag] = np.array(
                sorted(by_timestamp.keys()), dtype=np.int64
            )
            logger.info(
                f"Loaded {sum(len(v) for v in by_timestamp.values())} "
                f"Boxy 2D boxes for {tag} RGB"
            )

    def _read_2d_bounding_box_csv(self, path: Path) -> dict[int, list[Boxy2DBB]]:
        by_timestamp: dict[int, list[Boxy2DBB]] = {}
        with path.open(newline="") as fp:
            reader = csv.DictReader(fp, skipinitialspace=True)
            for row in reader:
                if row.get("stream_id") != "214-1":
                    continue
                try:
                    object_uid = int(row["object_uid"])
                    timestamp = int(row["timestamp[ns]"])
                    box = Boxy2DBB(
                        object_uid=object_uid,
                        xmin=max(0.0, float(row["x_min[pixel]"])),
                        xmax=max(0.0, float(row["x_max[pixel]"])),
                        ymin=max(0.0, float(row["y_min[pixel]"])),
                        ymax=max(0.0, float(row["y_max[pixel]"])),
                        visibility=float(row["visibility_ratio[%]"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(
                    np.isfinite(v)
                    for v in (box.xmin, box.xmax, box.ymin, box.ymax, box.visibility)
                ):
                    continue
                if box.visibility < _BBOX_VISIBILITY_THRESHOLD:
                    continue
                by_timestamp.setdefault(timestamp, []).append(box)
        return by_timestamp
