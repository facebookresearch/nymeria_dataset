# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from projectaria_tools.core.sophus import SE3


class BoundingBoxDataProvider:
    """
    Data provider for 3D bounding boxes loaded from Nymeria scene CSV files.

    Expected files in scene directory:
    - 3d_bounding_box.csv: AABB in local object frame (min/max coordinates)
    - scene_objects.csv: Object poses in world frame (position + quaternion)
    - instances.json: Object metadata (categories, names, etc.)
    """

    def __init__(self, scene_dir: str) -> None:
        scene_path = Path(scene_dir)
        if not scene_path.is_dir():
            logger.warning(
                f"Scene directory {scene_dir} not found, bbox provider disabled"
            )
            self.is_loaded = False
            return

        bbox_csv = scene_path / "3d_bounding_box.csv"
        objects_csv = scene_path / "scene_objects.csv"
        instances_json = scene_path / "instances.json"

        if not all([bbox_csv.exists(), objects_csv.exists(), instances_json.exists()]):
            logger.warning(
                f"Missing required scene files in {scene_dir}, bbox provider disabled"
            )
            self.is_loaded = False
            return

        logger.info(f"Loading bounding boxes from {scene_dir}")

        # Load CSV files
        self.df_bbox = pd.read_csv(bbox_csv, skipinitialspace=True)
        self.df_objects = pd.read_csv(objects_csv, skipinitialspace=True)

        # Load instances JSON
        with open(instances_json, "r") as f:
            self.instances = json.load(f)

        # Find objects with both bbox and pose data
        self.object_uids = sorted(
            set(self.df_bbox["object_uid"]) & set(self.df_objects["object_uid"])
        )
        self.num_boxes = len(self.object_uids)

        if self.num_boxes == 0:
            logger.warning("No objects found with both bbox and pose data")
            self.is_loaded = False
            return

        logger.info(f"Loaded {self.num_boxes} objects with bounding boxes")

        # Pre-compute bbox data (static boxes)
        self._precompute_bboxes()

        self.is_loaded = True

    def _precompute_bboxes(self) -> None:
        """Pre-compute bbox parameters for all objects (static scene)."""
        self.bbox_centers = np.zeros((self.num_boxes, 3), dtype=np.float32)
        self.bbox_dimensions = np.zeros((self.num_boxes, 3), dtype=np.float32)
        self.bbox_rotations = np.zeros((self.num_boxes, 4), dtype=np.float32)
        self.bbox_labels = []
        self.bbox_colors = np.zeros((self.num_boxes, 3), dtype=np.uint8)

        for box_idx, obj_uid in enumerate(self.object_uids):
            # Get AABB data (in local object frame)
            bbox_row = self.df_bbox[self.df_bbox["object_uid"] == obj_uid].iloc[0]
            xmin = bbox_row["p_local_obj_xmin[m]"]
            xmax = bbox_row["p_local_obj_xmax[m]"]
            ymin = bbox_row["p_local_obj_ymin[m]"]
            ymax = bbox_row["p_local_obj_ymax[m]"]
            zmin = bbox_row["p_local_obj_zmin[m]"]
            zmax = bbox_row["p_local_obj_zmax[m]"]

            # Get object pose (in world frame)
            obj_row = self.df_objects[self.df_objects["object_uid"] == obj_uid].iloc[0]
            t_wo_x = obj_row["t_wo_x[m]"]
            t_wo_y = obj_row["t_wo_y[m]"]
            t_wo_z = obj_row["t_wo_z[m]"]
            q_wo_w = obj_row["q_wo_w"]
            q_wo_x = obj_row["q_wo_x"]
            q_wo_y = obj_row["q_wo_y"]
            q_wo_z = obj_row["q_wo_z"]

            # Compute bbox dimensions
            dimensions = np.array(
                [
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ],
                dtype=np.float32,
            )

            # Center in world frame (object position)
            center_world = np.array([t_wo_x, t_wo_y, t_wo_z], dtype=np.float32)

            # Quaternion (w, x, y, z)
            rotation = np.array([q_wo_w, q_wo_x, q_wo_y, q_wo_z], dtype=np.float32)

            # Store bbox parameters
            self.bbox_centers[box_idx] = center_world
            self.bbox_dimensions[box_idx] = dimensions
            self.bbox_rotations[box_idx] = rotation

            # Get object metadata
            obj_instance = self.instances.get(str(obj_uid), {})
            category = obj_instance.get("category", "Unknown")
            instance_id = obj_instance.get("instance_id", obj_uid)
            label = f"{category}_{instance_id}"
            self.bbox_labels.append(label)

            # Generate color based on category
            category_uid = obj_instance.get("category_uid", 0)
            self.bbox_colors[box_idx] = self._generate_color(category_uid)

        self.bbox_labels = np.array(self.bbox_labels)

    def _generate_color(self, category_uid: int) -> np.ndarray:
        """Generate distinct color for a category."""
        hue = (category_uid * 30) % 360
        color = np.zeros(3, dtype=np.uint8)

        if hue < 60:
            color = [255, int(hue * 4.25), 0]
        elif hue < 120:
            color = [255 - int((hue - 60) * 4.25), 255, 0]
        elif hue < 180:
            color = [0, 255, int((hue - 120) * 4.25)]
        elif hue < 240:
            color = [0, 255 - int((hue - 180) * 4.25), 255]
        elif hue < 300:
            color = [int((hue - 240) * 4.25), 0, 255]
        else:
            color = [255, 0, 255 - int((hue - 300) * 4.25)]

        return np.array(color, dtype=np.uint8)

    @property
    def is_valid(self) -> bool:
        """Check if bbox data was loaded successfully."""
        return self.is_loaded

    def get_global_timespan_us(self) -> tuple[int, int]:
        """
        Get time range (static boxes, so returns (0, inf)).
        Scene bboxes are static and valid for all timestamps.
        """
        if not self.is_valid:
            return (0, 0)
        return (0, int(1e15))  # Valid for any timestamp

    def get_bboxes_at_timestamp(
        self, t_us: int, T_world_ref: SE3 = None
    ) -> tuple[np.ndarray, dict]:
        """
        Get bounding boxes at a specific timestamp.
        Since scene boxes are static, timestamp is ignored.

        Args:
            t_us: Query timestamp in microseconds (ignored for static boxes)
            T_world_ref: Optional SE3 transform to apply to all bboxes

        Returns:
            bbox_edges: (M, 12, 2, 3) array of edge lines for M boxes, 12 edges each
            bbox_info: dict with labels, colors, and other metadata
        """
        if not self.is_valid:
            return None, None

        # Convert to 3D wireframe edges
        bbox_edges = self._bboxes_to_edges(
            self.bbox_centers, self.bbox_dimensions, self.bbox_rotations, T_world_ref
        )

        # Prepare metadata
        bbox_info = {
            "labels": self.bbox_labels,
            "colors": self.bbox_colors,
            "num_boxes": self.num_boxes,
            "timestamp_us": t_us,
        }

        return bbox_edges, bbox_info

    def _bboxes_to_edges(
        self,
        centers: np.ndarray,
        dimensions: np.ndarray,
        rotations: np.ndarray,
        T_world_ref: SE3 = None,
    ) -> np.ndarray:
        """
        Convert bbox parameters to 3D wireframe edges.

        Args:
            centers: (M, 3) bbox centers in world frame
            dimensions: (M, 3) bbox dimensions (width, height, depth)
            rotations: (M, 4) quaternions (w, x, y, z)
            T_world_ref: Optional transform to apply

        Returns:
            edges: (M, 12, 2, 3) array of edge lines
        """
        M = centers.shape[0]
        edges = np.zeros((M, 12, 2, 3), dtype=np.float32)

        for i in range(M):
            # Get 8 corners of the bbox in local frame
            w, h, d = dimensions[i] / 2.0  # half dimensions

            # Define 8 corners (local coordinates, bbox-centered)
            corners_local = np.array(
                [
                    [-w, -h, -d],  # 0: back-bottom-left
                    [+w, -h, -d],  # 1: back-bottom-right
                    [+w, +h, -d],  # 2: back-top-right
                    [-w, +h, -d],  # 3: back-top-left
                    [-w, -h, +d],  # 4: front-bottom-left
                    [+w, -h, +d],  # 5: front-bottom-right
                    [+w, +h, +d],  # 6: front-top-right
                    [-w, +h, +d],  # 7: front-top-left
                ]
            )

            # Apply rotation and translation using SE3
            qw, qx, qy, qz = rotations[i]
            T_bbox = SE3.from_quat_and_translation(
                qw, np.array([qx, qy, qz]), centers[i]
            )

            # Apply optional world transform
            if T_world_ref is not None:
                T_bbox = T_world_ref @ T_bbox

            # Transform corners to world frame
            corners_world = np.zeros_like(corners_local)
            for j, corner in enumerate(corners_local):
                transformed = T_bbox @ corner.reshape(3, 1)
                corners_world[j] = transformed.squeeze()

            # Define 12 edges of the bbox (connecting corners)
            edge_indices = [
                # Bottom face (z = -d)
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                # Top face (z = +d)
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                # Vertical edges
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]

            for edge_idx, (start, end) in enumerate(edge_indices):
                edges[i, edge_idx, 0] = corners_world[start]
                edges[i, edge_idx, 1] = corners_world[end]

        return edges


def create_bbox_data_provider(scene_dir: str) -> BoundingBoxDataProvider | None:
    """
    Factory function to create a BoundingBoxDataProvider from scene directory.

    Args:
        scene_dir: Path to scene directory containing CSV and JSON files

    Returns:
        BoundingBoxDataProvider instance or None if loading fails
    """
    if Path(scene_dir).is_dir():
        provider = BoundingBoxDataProvider(scene_dir=scene_dir)
        if provider.is_valid:
            return provider
    return None
