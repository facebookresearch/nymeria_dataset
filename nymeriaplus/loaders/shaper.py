# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for ShapeR per-instance object meshes."""

from __future__ import annotations

import csv
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nymeriaplus.loaders.object_palette import (
    build_category_palette,
    CategoryPaletteEntry,
    generate_category_color,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShaperMesh:
    file_name: str
    object_uid: int
    variant_id: int
    score: int
    category: str
    category_uid: int
    color: tuple[float, float, float]
    vertices: np.ndarray
    faces: np.ndarray


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def _load_binary_little_endian_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as fp:
        header_lines: list[str] = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f"PLY header missing end_header: {path}")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"only binary_little_endian PLY is supported: {path}")

        vertex_count = 0
        face_count = 0
        for line in header_lines:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif len(parts) == 3 and parts[:2] == ["element", "face"]:
                face_count = int(parts[2])

        vertices = np.frombuffer(fp.read(vertex_count * 12), dtype="<f4").reshape(
            vertex_count, 3
        )
        faces: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            n = struct.unpack("<B", fp.read(1))[0]
            indices = struct.unpack(f"<{n}i", fp.read(n * 4))
            if n == 3:
                faces.append((indices[0], indices[1], indices[2]))
            elif n > 3:
                for i in range(1, n - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))

    return vertices.astype(np.float32), np.asarray(faces, dtype=np.uint32)


def _best_metadata_rows(metadata_csv: Path) -> dict[int, dict[str, str]]:
    with metadata_csv.open(newline="") as fp:
        reader = csv.DictReader(fp)
        best: dict[int, dict[str, str]] = {}
        for row in reader:
            object_uid = int(row["object_uid"])
            row["from_basemap"] = str(_parse_bool(row.get("from_basemap", "False")))
            current = best.get(object_uid)
            if current is None:
                best[object_uid] = row
                continue
            score = int(row["score"])
            current_score = int(current["score"])
            variant_id = int(row["variant_id"])
            current_variant_id = int(current["variant_id"])
            if score > current_score or (
                score == current_score and variant_id > current_variant_id
            ):
                best[object_uid] = row
        return best


class ShaperMeshLoader:
    """Load ShapeR meshes from ``objects/shaper`` and color by category."""

    def __init__(
        self,
        shaper_dir: Path,
        *,
        instances_path: Path | None = None,
        category_palette: dict[str, CategoryPaletteEntry] | None = None,
    ) -> None:
        self.shaper_dir = shaper_dir
        self.category_palette = category_palette or {}
        self.meshes: list[ShaperMesh] = []
        self._is_valid = False

        metadata_csv = shaper_dir / "shaper_metadata.csv"
        if not shaper_dir.is_dir() or not metadata_csv.is_file():
            logger.warning(f"ShapeR meshes unavailable in {shaper_dir}")
            return

        instances: dict[str, dict] = {}
        if instances_path is not None and instances_path.is_file():
            with instances_path.open() as fp:
                instances = json.load(fp)
            if not self.category_palette:
                self.category_palette = build_category_palette(instances)

        for object_uid, row in sorted(_best_metadata_rows(metadata_csv).items()):
            ply_path = shaper_dir / row["file_name"]
            if not ply_path.is_file():
                logger.warning(f"ShapeR mesh listed in metadata is missing: {ply_path}")
                continue
            vertices, faces = _load_binary_little_endian_ply(ply_path)
            instance = instances.get(str(object_uid), {})
            category = str(instance.get("category", row.get("category", "Unknown")))
            category_uid = int(instance.get("category_uid", 0))
            palette_entry = self.category_palette.get(category)
            if palette_entry is not None:
                color = palette_entry.color_float
                category_uid = palette_entry.category_uid
            else:
                color_u8 = generate_category_color(category_uid)
                color = tuple(float(v) / 255.0 for v in color_u8)
            self.meshes.append(
                ShaperMesh(
                    file_name=row["file_name"],
                    object_uid=object_uid,
                    variant_id=int(row["variant_id"]),
                    score=int(row["score"]),
                    category=category,
                    category_uid=category_uid,
                    color=color,
                    vertices=vertices,
                    faces=faces,
                )
            )

        self._is_valid = bool(self.meshes)
        if self._is_valid:
            logger.info(f"Loaded {len(self.meshes)} ShapeR object meshes")

    @property
    def is_valid(self) -> bool:
        return self._is_valid
