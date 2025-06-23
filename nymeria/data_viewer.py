# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger
from nymeria.data_provider import NymeriaDataProvider
from PIL import Image
from projectaria_tools.core.sensor_data import ImageData
from projectaria_tools.core.sophus import SE3
from tqdm import tqdm


@dataclass(frozen=True)
class ViewerConfig:
    output_rrd: Path = None
    sample_fps: float = 10
    rotate_rgb: bool = True
    downsample_rgb: bool = True
    jpeg_quality: int = 90
    traj_tail_length: int = 100

    ep_recording_head: str = "recording_head/2d"
    ep_recording_observer: str = "recording_observer/2d"

    point_radii: float = 0.008
    line_radii: float = 0.008
    skel_radii: float = 0.01


class NymeriaViewer(ViewerConfig):
    palette: dict[str, list] = {
        "recording_head": [255, 0, 0],
        "recording_lwrist": [0, 255, 0],
        "recording_rwrist": [0, 0, 255],
        "recording_observer": [61, 0, 118],
        "pointcloud": [128, 128, 128, 128],
        "momentum": [218, 234, 134],
    }
    color_skeleton = np.array(
        [
            [127, 0, 255],
            [105, 34, 254],
            [81, 71, 252],
            [59, 103, 249],
            [35, 136, 244],
            [11, 167, 238],
            [10, 191, 232],
            [34, 214, 223],
            [58, 232, 214],
            [80, 244, 204],
            [104, 252, 192],
            [128, 254, 179],
            [150, 252, 167],
            [174, 244, 152],
            [196, 232, 138],
            [220, 214, 122],
            [244, 191, 105],
            [255, 167, 89],
            [255, 136, 71],
            [255, 103, 53],
            [255, 71, 36],
            [255, 34, 17],
        ]
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="3d"),
            rrb.Vertical(
                rrb.Spatial2DView(name="2d participant", origin=self.ep_recording_head),
                rrb.Spatial2DView(
                    name="2d observer", origin=self.ep_recording_observer
                ),
            ),
        )

        rr.init(
            "nymeria data viewer",
            spawn=(self.output_rrd is None),
            recording_id=uuid4(),
            default_blueprint=blueprint,
        )
        if self.output_rrd is not None:
            rr.save(self.output_rrd)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._init_mesh: bool = False
        self._epaths_3d: set[str] = set()
        self._traj_deques: dict[str, deque] = {}

    def __call__(self, nymeria_dp: NymeriaDataProvider):
        # add static scene
        self.__log_pointcloud(nymeria_dp)
        self.__log_trajectory(nymeria_dp)

        # add dynamic scene
        t_ns_start, t_ns_end = nymeria_dp.timespan_ns
        dt: int = int(1e9 / self.sample_fps)
        for idx, t_ns in tqdm(enumerate(range(t_ns_start, t_ns_end, dt))):
            rr.set_time_sequence("frames", idx)
            rr.set_time_nanos("timestamps_ns", t_ns)

            self.__log_synced_video(t_ns, nymeria_dp)
            self.__log_synced_poses(t_ns, nymeria_dp)

            self.__set_viewpoint()

    def __log_pointcloud(self, nymeria_dp: NymeriaDataProvider) -> None:
        pointclouds = nymeria_dp.get_all_pointclouds()
        for tag, pts in pointclouds.items():
            logger.info(f"add point cloud {tag}")
            cc = self.palette.get("pointcloud")
            ep = f"world/semidense_pts/{tag}"
            rr.log(
                entity_path=ep,
                entity=rr.Points3D(pts, colors=cc, radii=self.point_radii),
                static=True,
            )
            self._epaths_3d.add(ep)

    def __log_trajectory(self, nymeria_dp: NymeriaDataProvider) -> None:
        trajs: dict[str, np.ndarray] = nymeria_dp.get_all_trajectories()
        for tag, traj in trajs.items():
            logger.info(f"add trajectory {tag}, {traj.shape=}")
            ep = f"world/traj_full/{tag}"
            rr.log(
                ep,
                rr.LineStrips3D(
                    traj[:, :3, 3], colors=self.palette.get(tag), radii=self.line_radii
                ),
                static=True,
            )
            self._epaths_3d.add(ep)

    def __log_synced_video(self, t_ns: int, nymeria_dp: NymeriaDataProvider) -> None:
        images: dict[str, tuple] = nymeria_dp.get_synced_rgb_videos(t_ns)
        for tag, data in images.items():
            rgb: ImageData = data[0]

            if self.downsample_rgb:
                rgb = rgb.to_numpy_array()[::2, ::2, :]
            rgb = Image.fromarray(rgb.astype(np.uint8))
            if self.rotate_rgb:
                rgb = rgb.rotate(-90)

            if tag in self.ep_recording_head:
                ep = self.ep_recording_head
            elif tag in self.ep_recording_observer:
                ep = self.ep_recording_observer
            rr.log(
                f"{ep}/214-1", rr.Image(rgb).compress(jpeg_quality=self.jpeg_quality)
            )

    def __log_synced_poses(self, t_ns: int, nymeria_dp: NymeriaDataProvider) -> None:
        poses: dict[str, any] = nymeria_dp.get_synced_poses(t_ns)

        self._T_mv: SE3 = None
        for tag, val in poses.items():
            if "recording" in tag and self.traj_tail_length > 0:
                traj = self._traj_deques.setdefault(tag, deque())
                if self.traj_tail_length > 0 and len(traj) == self.traj_tail_length:
                    traj.popleft()
                t = val.transform_world_device.translation()
                traj.append(t.squeeze().tolist())
                ep = f"world/traj_tail/{tag}"
                rr.log(
                    ep,
                    rr.LineStrips3D(
                        traj, colors=self.palette.get(tag), radii=self.line_radii
                    ),
                )
                self._epaths_3d.add(ep)

            if tag == "xsens":
                ep = "world/body/xsens_skel"
                logger.debug(f"xsens skeleton {val.shape = }")
                rr.log(
                    ep,
                    rr.LineStrips3D(
                        val, colors=self.color_skeleton, radii=self.skel_radii
                    ),
                    static=False,
                )
                self._epaths_3d.add(ep)
            if tag == "momentum":
                ep = "world/body/momentum_mesh"
                if self._init_mesh:
                    rr.log_components(ep, [rr.components.Position3DBatch(val)])
                else:
                    faces = nymeria_dp.body_dp.momentum_template_mesh.faces
                    normals = nymeria_dp.body_dp.momentum_template_mesh.normals
                    rr.log(
                        ep,
                        rr.Mesh3D(
                            triangle_indices=faces,
                            vertex_positions=val,
                            vertex_normals=normals,
                            vertex_colors=self.palette.get(tag),
                        ),
                    )
                    self._init_mesh = True
                self._epaths_3d.add(ep)

            if tag == "recording_head":
                self._T_mv = val.transform_world_device

    def __set_viewpoint(self, add_rotation: bool = False):
        if self._T_mv is None:
            return
        t = self._T_mv.translation() * -1.0
        Rz = np.eye(3)
        if add_rotation:
            R = self._T_mv.rotation().to_matrix()
            psi = np.arctan2(R[1, 0], R[0, 0])
            Rz[0:2, 0:2] = np.array(
                [np.cos(psi), -np.sin(psi), np.sin(psi), np.cos(psi)]
            ).reshape(2, 2)

        for ep in self._epaths_3d:
            rr.log(
                ep,
                rr.Transform3D(translation=t, mat3x3=Rz),
                static=False,
            )
