# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Single-file moderngl + imgui-bundle viewer for SynchronizedSequence data.

Run via the ``nymeriaplus-viewer`` CLI; this module's :func:`launch` is the
entry point invoked after data has been loaded and synchronized.

World frame: right-hand Z-up, meters. Same as the synced data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import moderngl
import moderngl_window as mglw
import numpy as np
from imgui_bundle import imgui
from moderngl_window.integrations.imgui_bundle import ModernglWindowRenderer
from nymeriaplus.data_loader import NymeriaPlusDataLoader
from nymeriaplus.loaders.mhr import MHRBodyLoader
from nymeriaplus.loaders.smpl import SMPLBodyLoader
from nymeriaplus.synchronized import SynchronizedSequence
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId

logger = logging.getLogger(__name__)

_RGB_STREAM_ID = StreamId("214-1")
_IMGUI_FONT_PATH = (
    Path(__file__).resolve().parent / "assets" / "JetBrainsMono-Regular.ttf"
)
_IMGUI_FONT_SIZE = 16.0
_PROJECTED_BBOX_EDGE_SAMPLE_SPACING_M = 0.2
_PROJECTED_BBOX_EDGE_MIN_SAMPLES = 10
_PROJECTED_BBOX_EDGE_MAX_SAMPLES = 20
_PROJECTED_BBOX_LINE_WIDTH = 1.5
_PROJECTED_BBOX_LABEL_SCALE = 1.0
_IMGUI_VERTEX_CAPACITY = 1_048_576
_IMGUI_INDEX_CAPACITY = 2_097_152


def _positive_float(value: object) -> float | None:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def _read_scale_value(value: object) -> float | None:
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    if isinstance(value, (tuple, list)):
        scales = [_positive_float(item) for item in value]
        valid_scales = [scale for scale in scales if scale is not None]
        return max(valid_scales) if valid_scales else None
    return _positive_float(value)


def _read_size(value: object) -> tuple[float, float] | None:
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    width = _positive_float(value[0])
    height = _positive_float(value[1])
    if width is None or height is None:
        return None
    return width, height


def _detect_ui_scale(window: Any) -> float:
    objects = [window]
    for name in ("_window", "window"):
        inner = getattr(window, name, None)
        if inner is not None:
            objects.append(inner)

    for obj in objects:
        for name in (
            "content_scale",
            "pixel_ratio",
            "scale",
            "dpi_scale",
            "get_content_scale",
            "get_pixel_ratio",
            "get_scale",
        ):
            scale = _read_scale_value(getattr(obj, name, None))
            if scale is not None:
                return scale

        size = _read_size(getattr(obj, "size", None))
        for buffer_name in ("buffer_size", "framebuffer_size"):
            buffer_size = _read_size(getattr(obj, buffer_name, None))
            if size is not None and buffer_size is not None:
                return max(buffer_size[0] / size[0], buffer_size[1] / size[1])

    return 1.0


@dataclass(frozen=True)
class ProjectedBox3D:
    object_uid: int
    points: list[np.ndarray]
    valid: list[np.ndarray]
    label_xy: tuple[float, float] | None


# SMPL skeleton kinematic tree (24 joints). Mirrors smplx default.
_SMPL_KINTREE = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

_XSENS_SEGMENT_PARENTS = [
    -1,  # pelvis
    0,  # l5
    1,  # l3
    2,  # t12
    3,  # t8
    4,  # neck
    5,  # head
    4,  # right shoulder
    7,  # right upper arm
    8,  # right forearm
    9,  # right hand
    4,  # left shoulder
    11,  # left upper arm
    12,  # left forearm
    13,  # left hand
    0,  # right upper leg
    15,  # right lower leg
    16,  # right foot
    17,  # right toe
    0,  # left upper leg
    19,  # left lower leg
    20,  # left foot
    21,  # left toe
]


# ---------------------------------------------------------------------------
# Shaders (GLSL 3.3)
# ---------------------------------------------------------------------------

_VS_MESH = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_normal_eye;
out vec3 v_world;
void main() {
    vec4 wp = vec4(in_pos, 1.0);
    v_world = in_pos;
    v_normal_eye = transpose(inverse(mat3(u_view))) * in_normal;
    gl_Position = u_proj * u_view * wp;
}
"""

_FS_MESH = """
#version 330
uniform vec3 u_color;
uniform vec3 u_light_dir;  // normalized, world space
uniform int u_shading;     // 0=normal, 1=flat
uniform float u_alpha;
uniform float u_normalshade;
in vec3 v_normal_eye;
in vec3 v_world;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal_eye);
    vec3 c;
    if (u_shading == 0) {
        c = 0.5 * n + 0.5;
    } else {
        // Match the HMD2 avatar renderer's softened body-color shading.
        float NdotL = (dot(n, u_light_dir) + 1.0) / 2.1;
        float lit = NdotL * u_normalshade + 1.0 - u_normalshade;
        c = u_color * lit;
    }
    frag = vec4(c, u_alpha);
}
"""

_VS_POINTS = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_point_size;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
    gl_PointSize = u_point_size;
}
"""

_FS_POINTS = """
#version 330
in vec3 v_color;
out vec4 frag;
void main() {
    // Round point sprites
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    frag = vec4(v_color, 1.0);
}
"""

_VS_LINES = """
#version 330
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
}
"""

_GS_LINES = """
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 u_viewport;
uniform float u_line_width;

in vec3 v_color[];
out vec3 g_color;

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 ndc0 = p0.xy / p0.w;
    vec2 ndc1 = p1.xy / p1.w;
    vec2 dir = (ndc1 - ndc0) * u_viewport;
    float len_dir = length(dir);
    if (len_dir < 1e-6) {
        dir = vec2(1.0, 0.0);
    } else {
        dir /= len_dir;
    }
    vec2 normal = vec2(-dir.y, dir.x);
    vec2 offset_ndc = normal * max(u_line_width, 1.0) / u_viewport;

    g_color = v_color[0];
    gl_Position = vec4(p0.xy + offset_ndc * p0.w, p0.z, p0.w);
    EmitVertex();
    gl_Position = vec4(p0.xy - offset_ndc * p0.w, p0.z, p0.w);
    EmitVertex();

    g_color = v_color[1];
    gl_Position = vec4(p1.xy + offset_ndc * p1.w, p1.z, p1.w);
    EmitVertex();
    gl_Position = vec4(p1.xy - offset_ndc * p1.w, p1.z, p1.w);
    EmitVertex();

    EndPrimitive();
}
"""

_FS_LINES = """
#version 330
in vec3 g_color;
out vec4 frag;
void main() {
    frag = vec4(g_color, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from axis (any length) and angle (radians)."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c = np.cos(angle)
    s = np.sin(angle)
    K = np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64
    )
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


class OrbitCamera:
    """Pangolin/vigl-style camera. Stores eye, target, up explicitly.

    Mouse mapping (vigl ``Handler3D``):
      - Left drag             -> pan (eye + target translate together)
      - Middle drag           -> rotate-in-place (target moves around eye)
      - Right drag            -> orbit (eye moves around target)
      - Left + Right drag     -> roll around the view direction
      - Scroll                -> dolly zoom (delta>0 zoom out)
    """

    ORBIT_SENSITIVITY: float = 0.01
    PAN_SENSITIVITY: float = 0.003
    ZOOM_FACTOR: float = 0.1

    def __init__(
        self,
        target: np.ndarray | None = None,
        distance: float = 4.0,
        yaw: float = 0.6,
        pitch: float = 0.4,
        fov_deg: float = 50.0,
        near: float = 0.05,
        far: float = 200.0,
        up: np.ndarray | None = None,
    ) -> None:
        if target is None:
            target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up is None:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.up = np.asarray(up, dtype=np.float64).copy()
        # Initial eye derived from yaw/pitch/distance for a comfortable starting view.
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        offset = np.array([cp * cy, cp * sy, sp], dtype=np.float64) * distance
        self.eye_pos = self.target + offset
        self.fov_deg = float(fov_deg)
        self.near = float(near)
        self.far = float(far)
        self._follow_prev: np.ndarray | None = None

    # -------------------------------------------------------------- matrices

    def view(self) -> np.ndarray:
        return look_at(
            self.eye_pos.astype(np.float32),
            self.target.astype(np.float32),
            self.up.astype(np.float32),
        )

    def proj(self, aspect: float) -> np.ndarray:
        return perspective(np.radians(self.fov_deg), aspect, self.near, self.far)

    # ---------------------------------------------------------------- helpers

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Camera-space right, up, forward (forward points toward target)."""
        f = self.target - self.eye_pos
        f /= np.linalg.norm(f) + 1e-12
        r = np.cross(f, self.up)
        r /= np.linalg.norm(r) + 1e-12
        u = np.cross(r, f)
        return r, u, f

    # ----------------------------------------------------------- public verbs

    def follow(self, T_world_device: np.ndarray) -> None:
        """Translate eye + target by the change in target position since last call.

        Matches vigl's `_apply_follow_camera`: only applies the *delta* in head
        position, so any user pan/orbit/zoom offsets persist frame-to-frame.
        """
        new_t = T_world_device[:3, 3].astype(np.float64)
        if self._follow_prev is not None:
            delta = new_t - self._follow_prev
            self.target += delta
            self.eye_pos += delta
        self._follow_prev = new_t.copy()

    def reset_follow(self) -> None:
        """Clear follow-tracking state. Call when follow toggles off so re-enabling
        doesn't teleport the camera by accumulated body movement.
        """
        self._follow_prev = None

    def pan(self, dx: float, dy: float) -> None:
        """Left-drag: translate eye + target along screen plane."""
        distance = float(np.linalg.norm(self.eye_pos - self.target))
        scale = distance * self.PAN_SENSITIVITY
        r, u, _ = self._basis()
        translation = -dx * scale * r + dy * scale * u
        self.target += translation
        self.eye_pos += translation

    def orbit(self, dx: float, dy: float) -> None:
        """Right-drag: orbit eye around target."""
        offset = self.eye_pos - self.target
        # Yaw around world up.
        R_yaw = _rodrigues(self.up, -dx * self.ORBIT_SENSITIVITY)
        offset = R_yaw @ offset
        # Pitch around camera right.
        r, _, _ = self._basis_from_eye(self.target + offset)
        R_pitch = _rodrigues(r, -dy * self.ORBIT_SENSITIVITY)
        new_offset = R_pitch @ offset
        # Avoid pole-flip.
        n = new_offset / (np.linalg.norm(new_offset) + 1e-12)
        if abs(np.dot(n, self.up / np.linalg.norm(self.up))) > 0.99:
            new_offset = offset
        self.eye_pos = self.target + new_offset

    def rotate_in_place(self, dx: float, dy: float) -> None:
        """Middle-drag: rotate target around eye (camera stays fixed)."""
        offset = self.target - self.eye_pos
        distance = float(np.linalg.norm(offset))
        R_yaw = _rodrigues(self.up, -dx * self.ORBIT_SENSITIVITY)
        offset = R_yaw @ offset
        r, _, _ = self._basis_from_eye(self.eye_pos + offset)
        R_pitch = _rodrigues(r, -dy * self.ORBIT_SENSITIVITY)
        offset = R_pitch @ offset
        offset = offset / (np.linalg.norm(offset) + 1e-12) * distance
        self.target = self.eye_pos + offset

    def roll(self, dx: float, dy: float) -> None:
        """Left+Right drag: roll camera up around the view direction."""
        _, u, f = self._basis()
        R = _rodrigues(f, dx * self.ORBIT_SENSITIVITY)
        new_up = R @ u
        # Project new_up onto the plane perpendicular to forward to keep it valid.
        new_up -= np.dot(new_up, f) * f
        n = np.linalg.norm(new_up)
        if n > 1e-9:
            self.up = new_up / n

    def zoom(self, delta: float) -> None:
        """Scroll: dolly. delta>0 -> zoom out, delta<0 -> zoom in."""
        direction = self.eye_pos - self.target
        distance = float(np.linalg.norm(direction))
        factor = max(1.0 + delta * self.ZOOM_FACTOR, 0.01)
        new_distance = float(np.clip(distance * factor, 0.05, 500.0))
        self.eye_pos = self.target + (direction / (distance + 1e-12)) * new_distance

    # --------------------------------------------------------------- internal

    def _basis_from_eye(
        self, eye: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self.target - eye
        f /= np.linalg.norm(f) + 1e-12
        r = np.cross(f, self.up)
        r /= np.linalg.norm(r) + 1e-12
        u = np.cross(r, f)
        return r, u, f


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2.0 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals from triangle mesh, summed with face areas."""
    n = np.zeros_like(verts)
    tri = verts[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    fn = np.cross(e1, e2)
    np.add.at(n, faces[:, 0], fn)
    np.add.at(n, faces[:, 1], fn)
    np.add.at(n, faces[:, 2], fn)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (n / norms).astype(np.float32)


def build_grid(
    size: float, divisions: int, color: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    half = size / 2.0
    step = size / divisions
    pts = []
    for i in range(divisions + 1):
        t = -half + i * step
        pts.append([[t, -half, 0.0], [t, half, 0.0]])
        pts.append([[-half, t, 0.0], [half, t, 0.0]])
    arr = np.array(pts, dtype=np.float32).reshape(-1, 3)
    cols = np.tile(np.array(color, dtype=np.float32), (arr.shape[0], 1))
    return arr, cols


def build_axes(scale: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(
        [
            [0, 0, 0],
            [scale, 0, 0],
            [0, 0, 0],
            [0, scale, 0],
            [0, 0, 0],
            [0, 0, scale],
        ],
        dtype=np.float32,
    )
    cols = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return pts, cols


def _frustum_vertices_local(scale: float, fov: float, aspect: float) -> np.ndarray:
    """Wireframe frustum in device-local frame (RDF: +Z forward, +Y down)."""
    h = scale * np.tan(fov / 2.0)
    w = h * aspect
    z = scale
    apex = np.array([0, 0, 0], dtype=np.float32)
    corners = np.array(
        [[-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]], dtype=np.float32
    )
    edges = []
    for c in corners:
        edges.append([apex, c])
    for i in range(4):
        edges.append([corners[i], corners[(i + 1) % 4]])
    return np.array(edges, dtype=np.float32).reshape(-1, 3)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    out = (T @ homog.T).T
    return out[:, :3].astype(np.float32)


def build_skeleton_segments(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(c, p) for c, p in enumerate(_SMPL_KINTREE) if p >= 0]
    segs = np.array([[joints[c], joints[p]] for c, p in pairs], dtype=np.float32)
    n_segs = segs.shape[0]
    cols = np.zeros((n_segs, 2, 3), dtype=np.float32)
    for i in range(n_segs):
        t = i / max(n_segs - 1, 1)
        cols[i] = np.array([t, 1.0 - t, 0.5])
    return segs.reshape(-1, 3), cols.reshape(-1, 3)


def build_xsens_segments(
    segment_positions: np.ndarray, color: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (child, parent)
        for child, parent in enumerate(_XSENS_SEGMENT_PARENTS)
        if parent >= 0 and child < segment_positions.shape[0]
    ]
    segs = np.array(
        [
            [segment_positions[parent], segment_positions[child]]
            for child, parent in pairs
        ],
        dtype=np.float32,
    )
    cols = np.tile(np.array(color, dtype=np.float32), (segs.shape[0], 2, 1))
    return segs.reshape(-1, 3), cols.reshape(-1, 3)


def build_parented_skeleton_segments(
    joints: np.ndarray,
    parents: np.ndarray,
    color: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (child, int(parent))
        for child, parent in enumerate(parents)
        if parent >= 0
        and child < joints.shape[0]
        and parent < joints.shape[0]
        and not (parent == 0 and child == 1)
    ]
    if not pairs:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    segs = np.array([[joints[parent], joints[child]] for child, parent in pairs])
    cols = np.tile(np.array(color, dtype=np.float32), (segs.shape[0], 2, 1))
    return segs.reshape(-1, 3).astype(np.float32), cols.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class MeshRenderer:
    """SMPL mesh: positions + normals VBO, fixed face IBO."""

    def __init__(self, ctx: moderngl.Context, n_verts: int, faces: np.ndarray) -> None:
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VS_MESH, fragment_shader=_FS_MESH)
        self.vbo_pos = ctx.buffer(reserve=n_verts * 12, dynamic=True)
        self.vbo_nrm = ctx.buffer(reserve=n_verts * 12, dynamic=True)
        self.ibo = ctx.buffer(faces.astype(np.uint32).tobytes())
        self.vao = ctx.vertex_array(
            self.prog,
            [
                (self.vbo_pos, "3f", "in_pos"),
                (self.vbo_nrm, "3f", "in_normal"),
            ],
            index_buffer=self.ibo,
        )
        self.n_indices = faces.size

    def update(self, verts: np.ndarray, normals: np.ndarray) -> None:
        self.vbo_pos.write(verts.astype(np.float32).tobytes())
        self.vbo_nrm.write(normals.astype(np.float32).tobytes())

    def draw(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        color: tuple[float, float, float],
        shading: int,
        wireframe: bool,
        alpha: float = 1.0,
    ) -> None:
        self.prog["u_view"].write(view.T.tobytes())
        self.prog["u_proj"].write(proj.T.tobytes())
        self.prog["u_color"].value = color
        self.prog["u_shading"].value = shading
        self.prog["u_light_dir"].value = (0.41, 0.82, 0.41)
        self.prog["u_alpha"].value = float(alpha)
        self.prog["u_normalshade"].value = 0.7

        # ImGui and other passes can leave GL state behind. Match vigl's mesh
        # renderer by relying on depth testing instead of forced face culling,
        # and always restore depth writes before drawing the body. If depth
        # writes are off, later SMPL triangles can show through front surfaces.
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.CULL_FACE)

        translucent = alpha < 0.999
        if translucent:
            # For translucent SMPL, draw only the front-facing surface and do
            # not write depth. This keeps the scene visible through the body
            # while avoiding front/back triangle accumulation inside the mesh.
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.front_face = "ccw"
            self.ctx.cull_face = "back"
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.depth_mask = False
        else:
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.disable(moderngl.BLEND)

        if wireframe:
            self.ctx.wireframe = True
            self.vao.render(moderngl.TRIANGLES)
            self.ctx.wireframe = False
        else:
            self.vao.render(moderngl.TRIANGLES)

        self.ctx.depth_mask = True
        self.ctx.depth_func = "<="


class StaticMeshBatchRenderer:
    """Static object meshes with one uniform color per mesh."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VS_MESH, fragment_shader=_FS_MESH)
        self.meshes: list[dict] = []

    def add_mesh(
        self,
        verts: np.ndarray,
        normals: np.ndarray,
        faces: np.ndarray,
        color: tuple[float, float, float],
    ) -> None:
        vbo_pos = self.ctx.buffer(verts.astype(np.float32).tobytes())
        vbo_nrm = self.ctx.buffer(normals.astype(np.float32).tobytes())
        ibo = self.ctx.buffer(faces.astype(np.uint32).tobytes())
        vao = self.ctx.vertex_array(
            self.prog,
            [
                (vbo_pos, "3f", "in_pos"),
                (vbo_nrm, "3f", "in_normal"),
            ],
            index_buffer=ibo,
        )
        self.meshes.append(
            {
                "vao": vao,
                "vbo_pos": vbo_pos,
                "vbo_nrm": vbo_nrm,
                "ibo": ibo,
                "n_indices": int(faces.size),
                "color": color,
            }
        )

    def draw(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        wireframe: bool,
        alpha: float = 1.0,
    ) -> None:
        if not self.meshes:
            return
        self.prog["u_view"].write(view.T.tobytes())
        self.prog["u_proj"].write(proj.T.tobytes())
        self.prog["u_shading"].value = 1
        self.prog["u_light_dir"].value = (0.41, 0.82, 0.41)
        self.prog["u_alpha"].value = float(alpha)
        self.prog["u_normalshade"].value = 0.7
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.CULL_FACE)

        translucent = alpha < 0.999
        if translucent:
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.front_face = "ccw"
            self.ctx.cull_face = "back"
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.depth_mask = False
        else:
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.disable(moderngl.BLEND)

        self.ctx.wireframe = wireframe
        for mesh in self.meshes:
            self.prog["u_color"].value = mesh["color"]
            mesh["vao"].render(moderngl.TRIANGLES)
        self.ctx.wireframe = False
        self.ctx.depth_mask = True
        self.ctx.depth_func = "<="


class PointsRenderer:
    def __init__(self, ctx: moderngl.Context, max_points: int) -> None:
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VS_POINTS, fragment_shader=_FS_POINTS)
        self.max_points = max_points
        self.vbo_pos = ctx.buffer(reserve=max_points * 12, dynamic=True)
        self.vbo_col = ctx.buffer(reserve=max_points * 12, dynamic=True)
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo_pos, "3f", "in_pos"), (self.vbo_col, "3f", "in_color")],
        )
        self.count = 0

    def update(self, pts: np.ndarray, colors: np.ndarray) -> None:
        n = pts.shape[0]
        if n > self.max_points:
            self.vbo_pos.orphan(size=n * 12)
            self.vbo_col.orphan(size=n * 12)
            self.max_points = n
        self.vbo_pos.write(pts.astype(np.float32).tobytes())
        self.vbo_col.write(colors.astype(np.float32).tobytes())
        self.count = n

    def update_colors(self, colors: np.ndarray) -> None:
        if colors.shape[0] != self.count:
            raise ValueError(
                f"color count mismatch: got {colors.shape[0]}, expected {self.count}"
            )
        self.vbo_col.write(colors.astype(np.float32).tobytes())

    def draw(self, view: np.ndarray, proj: np.ndarray, point_size: float) -> None:
        if self.count == 0:
            return
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.prog["u_view"].write(view.T.tobytes())
        self.prog["u_proj"].write(proj.T.tobytes())
        self.prog["u_point_size"].value = point_size
        self.vao.render(moderngl.POINTS, vertices=self.count)


class LinesRenderer:
    def __init__(self, ctx: moderngl.Context, max_verts: int) -> None:
        self.ctx = ctx
        self.prog = ctx.program(
            vertex_shader=_VS_LINES,
            geometry_shader=_GS_LINES,
            fragment_shader=_FS_LINES,
        )
        self.max_verts = max_verts
        self.vbo_pos = ctx.buffer(reserve=max_verts * 12, dynamic=True)
        self.vbo_col = ctx.buffer(reserve=max_verts * 12, dynamic=True)
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo_pos, "3f", "in_pos"), (self.vbo_col, "3f", "in_color")],
        )
        self.count = 0

    def update(self, pts: np.ndarray, colors: np.ndarray) -> None:
        n = pts.shape[0]
        if n > self.max_verts:
            self.vbo_pos.orphan(size=n * 12)
            self.vbo_col.orphan(size=n * 12)
            self.max_verts = n
        self.vbo_pos.write(pts.astype(np.float32).tobytes())
        self.vbo_col.write(colors.astype(np.float32).tobytes())
        self.count = n

    def draw(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        viewport_size: tuple[int, int],
        line_width: float = 1.0,
    ) -> None:
        if self.count == 0:
            return
        self.prog["u_view"].write(view.T.tobytes())
        self.prog["u_proj"].write(proj.T.tobytes())
        self.prog["u_viewport"].value = (
            float(max(viewport_size[0], 1)),
            float(max(viewport_size[1], 1)),
        )
        self.prog["u_line_width"].value = float(line_width)
        self.vao.render(moderngl.LINES, vertices=self.count)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class NymeriaPlusViewer(mglw.WindowConfig):
    gl_version = (3, 3)
    samples = 4  # 4x MSAA on the default framebuffer (lines/edges/silhouettes)
    log_level = logging.WARNING  # quiet moderngl-window's startup chatter
    title = "NymeriaPlus Viewer"
    window_size = (1920, 1080)
    resizable = True
    aspect_ratio = None
    resource_dir = Path(__file__).parent
    splitter_hit_px = 6
    controls_min_w = 220
    controls_max_w = 420
    rgb_min_w = 240
    rgb_max_w = 768

    # Injected by launch() before init.
    _loader: NymeriaPlusDataLoader | None = None
    _synced: SynchronizedSequence | None = None
    _smpl: SMPLBodyLoader | None = None
    _mhr: MHRBodyLoader | None = None
    _ui_scale: float | None = None

    TRAJ_COLORS = {
        "head": (1.0, 0.31, 0.31),
        "lwrist": (0.31, 1.0, 0.31),
        "rwrist": (0.31, 0.31, 1.0),
        "observer": (0.71, 0.40, 0.85),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self._loader is not None and self._synced is not None
        loader, synced = self._loader, self._synced
        self.loader = loader
        self.synced = synced
        self.smpl = self._smpl
        self.mhr = self._mhr

        # imgui
        imgui.create_context()
        configured_ui_scale = _positive_float(self._ui_scale)
        self.ui_scale = (
            configured_ui_scale
            if configured_ui_scale is not None
            else _detect_ui_scale(self.wnd)
        )
        imgui.get_style().scale_all_sizes(self.ui_scale)
        if _IMGUI_FONT_PATH.is_file():
            imgui.get_io().fonts.add_font_from_file_ttf(
                str(_IMGUI_FONT_PATH), _IMGUI_FONT_SIZE * self.ui_scale
            )
        else:
            logger.warning(f"ImGui font missing: {_IMGUI_FONT_PATH}")
        self.imgui = ModernglWindowRenderer(self.wnd)
        self._reserve_imgui_buffers()

        # Dynamic body meshes.
        self.smpl_mesh_r: MeshRenderer | None = None
        if self.smpl is not None:
            try:
                faces = np.asarray(self.smpl.faces).astype(np.uint32)
                self.smpl_mesh_r = MeshRenderer(self.ctx, 6890, faces)
            except Exception as e:
                logger.warning(f"SMPL mesh disabled: {e}")
                self.smpl = None

        self.mhr_mesh_r: MeshRenderer | None = None
        if self.mhr is not None:
            faces = np.asarray(self.mhr.faces).astype(np.uint32)
            self.mhr_mesh_r = MeshRenderer(self.ctx, self.mhr.num_vertices, faces)
        self.object_mesh_r = StaticMeshBatchRenderer(self.ctx)
        if loader.mesh is not None:
            for mesh in loader.mesh.meshes:
                normals = compute_normals(mesh.vertices, mesh.faces.astype(np.int64))
                self.object_mesh_r.add_mesh(
                    mesh.vertices,
                    normals,
                    mesh.faces,
                    mesh.color,
                )

        # Renderers for dynamic primitives
        self.pcd_r = PointsRenderer(self.ctx, max_points=300_000)
        self.joints_r = PointsRenderer(self.ctx, max_points=128)
        self.lines_scene_r = LinesRenderer(
            self.ctx, max_verts=200_000
        )  # traj + frustums
        self.lines_skel_r = LinesRenderer(self.ctx, max_verts=128)
        self.lines_mhr_skel_r = LinesRenderer(self.ctx, max_verts=256)
        self.lines_xsens_r = LinesRenderer(self.ctx, max_verts=128)
        self.lines_static_r = LinesRenderer(self.ctx, max_verts=20_000)  # axes
        self.lines_bbox_r = LinesRenderer(self.ctx, max_verts=20_000)

        # Static scene scaffolding
        axes_pts, axes_cols = build_axes(0.5)
        self._static_pts = axes_pts
        self._static_cols = axes_cols
        self.lines_static_r.update(axes_pts, axes_cols)
        if loader.bbox is not None:
            self.lines_bbox_r.update(loader.bbox.line_points, loader.bbox.line_colors)

        # Point cloud (load + filter once)
        self._pcd_loaded = False

        # Trajectory positions extracted once (synced rate; no need to hit VRS)
        self._traj_positions: dict[str, np.ndarray] = {}
        for tag, attr in (
            ("head", "T_world_head"),
            ("lwrist", "T_world_lwrist"),
            ("rwrist", "T_world_rwrist"),
            ("observer", "T_world_observer"),
        ):
            arr = getattr(synced, attr, None)
            if arr is not None:
                self._traj_positions[tag] = arr[:, :3, 3].astype(np.float32).copy()

        self.n_frames = synced.num_frames

        # GUI state
        self.state: dict = {
            "frame": 0,
            "playing": False,
            "play_stride": 1,
            "follow": True,
            "show_smpl_mesh": self.smpl_mesh_r is not None,
            "show_mhr_mesh": self.mhr_mesh_r is not None,
            "wireframe": False,
            "shading_mode": 0,  # 0=normal, 1=flat
            "smpl_mesh_color": (0.51, 0.71, 0.90),
            "mhr_mesh_color": (0.90, 0.56, 0.30),
            "mesh_alpha": 0.8,
            "bg_value": 1.0,
            "show_smpl_skeleton": self.smpl is not None
            and synced.smpl_body_pose is not None,
            "show_mhr_skeleton": self.mhr is not None,
            "show_xsens_skeleton": synced.xsens_segment_positions is not None,
            "skeleton_line_width": 3.0,
            "joint_size": 12.0,
            "show_axes": True,
            "show_traj": True,
            "camera_line_width": 1.5,
            "trail_seconds": 5.0,
            "show_frustums": True,
            "show_bbox": loader.bbox is not None,
            "show_bbox_labels": loader.bbox is not None,
            "show_object_meshes": loader.mesh is not None,
            "object_mesh_alpha": 1.0,
            "show_pcd": True,
            "pcd_point_size": 1.5,
            "pcd_color": (0.71, 0.71, 0.71),
            "pcd_dep": 0.02,
            "pcd_invdep": 0.0004,
            "_pcd_dirty": True,
            "show_rgb": True,
            "show_projected_bbox": loader.bbox is not None
            and bool(loader.bbox.bb2d_by_recording),
            "show_projected_bbox_labels": loader.bbox is not None
            and bool(loader.bbox.bb2d_by_recording),
            "rgb_panel_w": 480,
        }

        # RGB textures (lazy-create to actual image sizes)
        self.rgb_textures: dict[str, moderngl.Texture | None] = {
            "head": None,
            "observer": None,
        }
        self.rgb_sizes: dict[str, tuple[int, int]] = {
            "head": (0, 0),
            "observer": (0, 0),
        }
        self.rgb_capture_timestamps_ns: dict[str, int | None] = {
            "head": None,
            "observer": None,
        }
        self.rgb_projected_boxes: dict[str, list[ProjectedBox3D]] = {
            "head": [],
            "observer": [],
        }
        self.rgb_source_heights: dict[str, int] = {"head": 0, "observer": 0}
        self.rgb_camera_calibs: dict[str, Any | None] = {
            "head": None,
            "observer": None,
        }
        self._last_rgb_idx: dict[str, int] = {"head": -1, "observer": -1}

        # Frame index tracking for VBO updates
        self._last_frame_idx = -1
        self._controls_w_user: int | None = None
        self._rgb_w_user: int | None = None
        self._active_splitter: str | None = None

        # Camera
        first_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if "head" in self._traj_positions:
            first_target = self._traj_positions["head"][0].copy()
        self.camera = OrbitCamera(target=first_target, distance=4.0)

        # GL state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)

    # -------------------------------------------------------------- helpers

    def _reserve_imgui_buffers(self) -> None:
        self.imgui._vertex_buffer.release()
        self.imgui._index_buffer.release()
        self.imgui._vao.release()
        self.imgui._vertex_buffer = self.ctx.buffer(
            reserve=imgui.VERTEX_SIZE * _IMGUI_VERTEX_CAPACITY
        )
        self.imgui._index_buffer = self.ctx.buffer(
            reserve=imgui.INDEX_SIZE * _IMGUI_INDEX_CAPACITY
        )
        self.imgui._vao = self.ctx.vertex_array(
            self.imgui._prog,
            [(self.imgui._vertex_buffer, "2f 2f 4f1", "Position", "UV", "Color")],
            index_buffer=self.imgui._index_buffer,
            index_element_size=imgui.INDEX_SIZE,
        )

    def _load_pcd_from_loader(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Pull semidense pointcloud from the head recording, filter."""
        rec = self.loader.recordings.get("recording_head")
        if rec is None or not rec.has_pointcloud:
            return None
        s = self.state
        pts = rec.get_pointcloud(
            th_invdep=s["pcd_invdep"],
            th_dep=s["pcd_dep"],
            max_point_count=None,
        )
        n = pts.shape[0]
        cols = np.tile(np.array(s["pcd_color"], dtype=np.float32), (n, 1))
        return pts.astype(np.float32), cols

    def _update_rgb(self, frame_idx: int) -> None:
        self._update_rgb_texture(frame_idx, "head", "recording_head")
        self._update_rgb_texture(frame_idx, "observer", "recording_observer")

    def _update_rgb_texture(self, frame_idx: int, tag: str, recording_key: str) -> None:
        if frame_idx == self._last_rgb_idx[tag]:
            return
        rec = self.loader.recordings.get(recording_key)
        if rec is None or not rec.has_rgb:
            return
        ts = self.synced.timestamps_ns
        if ts is None:
            return
        try:
            t_ns = int(ts[frame_idx])
            image_data, meta, _tdiff = rec.get_rgb_image(t_ns, TimeDomain.TIME_CODE)
            arr = image_data.to_numpy_array().astype(np.uint8)
            self.rgb_capture_timestamps_ns[tag] = int(meta.capture_timestamp_ns)
            boxes: list[ProjectedBox3D] = []
            if self.state["show_projected_bbox"]:
                bbox = self.loader.bbox
                if bbox is not None:
                    boxes = self._project_visible_3d_bboxes(
                        tag,
                        recording_key,
                        rec,
                        int(meta.capture_timestamp_ns),
                        arr.shape,
                    )
            src_h = arr.shape[0]
            self.rgb_projected_boxes[tag] = boxes
            self.rgb_source_heights[tag] = src_h
            arr = np.rot90(arr, k=-1).copy()  # Aria RGB rotated -90
            h, w = arr.shape[:2]
            if (w, h) != self.rgb_sizes[tag]:
                texture = self.rgb_textures[tag]
                if texture is not None:
                    self.imgui.remove_texture(texture)
                    texture.release()
                texture = self.ctx.texture((w, h), 3, arr.tobytes())
                texture.repeat_x = False
                texture.repeat_y = False
                self.imgui.register_texture(texture)
                self.rgb_textures[tag] = texture
                self.rgb_sizes[tag] = (w, h)
            else:
                texture = self.rgb_textures[tag]
                if texture is not None:
                    texture.write(arr.tobytes())
            self._last_rgb_idx[tag] = frame_idx
        except Exception as e:
            logger.debug(f"{tag} RGB read failed at frame {frame_idx}: {e}")

    def _nearest_visible_bbox_object_uids(self, tag: str) -> set[int]:
        bbox = self.loader.bbox
        if bbox is None:
            return set()
        by_timestamp = bbox.bb2d_by_recording.get(tag)
        timestamps = bbox.bb2d_timestamps.get(tag)
        query_ts = self.rgb_capture_timestamps_ns.get(tag)
        if by_timestamp is None or timestamps is None or query_ts is None:
            return set()
        idx = int(np.searchsorted(timestamps, int(query_ts)))
        candidates = []
        if idx < len(timestamps):
            candidates.append(int(timestamps[idx]))
        if idx > 0:
            candidates.append(int(timestamps[idx - 1]))
        if not candidates:
            return set()
        nearest_ts = min(candidates, key=lambda ts: abs(ts - int(query_ts)))
        if abs(nearest_ts - int(query_ts)) > 50_000_000:
            return set()
        return {box.object_uid for box in by_timestamp.get(nearest_ts, [])}

    def _rgb_camera_calib(
        self,
        tag: str,
        rec: Any,
        image_shape: tuple[int, ...],
    ) -> Any | None:
        calib = self.rgb_camera_calibs.get(tag)
        if calib is None:
            if rec.vrs is None:
                return None
            stream_label = rec.vrs.get_label_from_stream_id(_RGB_STREAM_ID)
            device_calib = rec.vrs.get_device_calibration()
            calib = device_calib.get_camera_calib(stream_label)

        height, width = image_shape[:2]
        calib_w, calib_h = (int(v) for v in calib.get_image_size())
        if (calib_w, calib_h) != (width, height):
            scale_w = width / calib_w
            scale_h = height / calib_h
            if not math.isclose(scale_w, scale_h, rel_tol=1e-4, abs_tol=1e-4):
                logger.debug(
                    f"{tag} RGB calibration size {(calib_w, calib_h)} does not "
                    f"match image size {(width, height)}"
                )
                return None
            calib = calib.rescale(
                np.array([width, height], dtype=np.int32),
                float(scale_w),
            )

        self.rgb_camera_calibs[tag] = calib
        return calib

    def _project_visible_3d_bboxes(  # noqa: C901
        self,
        tag: str,
        recording_key: str,
        rec: Any,
        capture_timestamp_ns: int,
        image_shape: tuple[int, ...],
    ) -> list[ProjectedBox3D]:
        bbox = self.loader.bbox
        if bbox is None or bbox.edges.size == 0:
            return []

        visible_uids = self._nearest_visible_bbox_object_uids(tag)
        if not visible_uids:
            return []

        calib = self._rgb_camera_calib(tag, rec, image_shape)
        if calib is None:
            return []

        try:
            pose, _tdiff = rec.get_pose(capture_timestamp_ns, TimeDomain.DEVICE_TIME)
        except Exception as e:
            logger.debug(f"{recording_key} pose unavailable for RGB boxes: {e}")
            return []

        T_world_device = pose.transform_world_device.to_matrix().astype(np.float64)
        T_device_camera = (
            calib.get_transform_device_camera().to_matrix().astype(np.float64)
        )
        T_camera_world = np.linalg.inv(T_world_device @ T_device_camera)
        height, width = image_shape[:2]

        projected: list[ProjectedBox3D] = []
        for object_idx, object_uid in enumerate(bbox.object_uids):
            if object_uid not in visible_uids:
                continue

            edge_points_2d: list[np.ndarray] = []
            edge_valid: list[np.ndarray] = []
            edges = bbox.edges[object_idx].astype(np.float64)
            for p0, p1 in edges:
                edge_length = float(np.linalg.norm(p1 - p0))
                sample_count = int(
                    np.clip(
                        math.ceil(edge_length / _PROJECTED_BBOX_EDGE_SAMPLE_SPACING_M)
                        + 1,
                        _PROJECTED_BBOX_EDGE_MIN_SAMPLES,
                        _PROJECTED_BBOX_EDGE_MAX_SAMPLES,
                    )
                )
                samples = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
                projected_edge = np.zeros((sample_count, 2), dtype=np.float32)
                valid_edge = np.zeros(sample_count, dtype=bool)
                points_world = (
                    p0[None, :] * (1.0 - samples[:, None])
                    + p1[None, :] * samples[:, None]
                )
                points_world_h = np.concatenate(
                    [points_world, np.ones((len(samples), 1), dtype=np.float64)],
                    axis=1,
                )
                points_camera = (T_camera_world @ points_world_h.T).T[:, :3]
                for sample_idx, point_camera in enumerate(points_camera):
                    if point_camera[2] <= 0.0:
                        continue
                    pixel = calib.project(point_camera)
                    if pixel is None:
                        continue
                    x, y = float(pixel[0]), float(pixel[1])
                    if not (0.0 <= x < width and 0.0 <= y < height):
                        continue
                    projected_edge[sample_idx] = (x, y)
                    valid_edge[sample_idx] = True
                edge_points_2d.append(projected_edge)
                edge_valid.append(valid_edge)

            label_xy = None
            valid_points_by_edge = [
                edge_pts[edge_mask]
                for edge_pts, edge_mask in zip(edge_points_2d, edge_valid, strict=False)
                if edge_mask.any()
            ]
            valid_points = (
                np.concatenate(valid_points_by_edge, axis=0)
                if valid_points_by_edge
                else np.zeros((0, 2), dtype=np.float32)
            )
            if len(valid_points) > 0:
                label_xy = (
                    float(valid_points[:, 0].mean()),
                    float(valid_points[:, 1].mean()),
                )
                projected.append(
                    ProjectedBox3D(
                        object_uid=object_uid,
                        points=edge_points_2d,
                        valid=edge_valid,
                        label_xy=label_xy,
                    )
                )

        return projected

    def _build_dynamic_lines(self, frame_idx: int) -> None:
        """Compose trajectories + skeleton + frustums into one VBO."""
        s = self.state
        synced = self.synced
        scene_pts: list[np.ndarray] = []
        scene_cols: list[np.ndarray] = []

        # Trajectories (trailing window)
        if s["show_traj"]:
            trail_frames = int(s["trail_seconds"] * synced.fps)
            for tag, positions in self._traj_positions.items():
                end = min(frame_idx + 1, positions.shape[0])
                start = max(0, end - trail_frames)
                if end - start < 2:
                    continue
                pts = positions[start:end]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 3)
                base = np.array(self.TRAJ_COLORS[tag], dtype=np.float32)
                bg_v = float(self.state["bg_value"])
                bg = np.array([bg_v, bg_v, bg_v], dtype=np.float32)
                n = segs.shape[0]
                alpha = np.linspace(0.05, 1.0, n, dtype=np.float32)
                seg_cols = (
                    bg[None, :] * (1.0 - alpha[:, None])
                    + base[None, :] * alpha[:, None]
                )
                seg_cols = np.repeat(seg_cols[:, None, :], 2, axis=1)  # (N, 2, 3)
                scene_pts.append(segs.reshape(-1, 3))
                scene_cols.append(seg_cols.reshape(-1, 3))

        # Skeleton bones
        if (
            s["show_smpl_skeleton"]
            and self.smpl_mesh_r is not None
            and self._joints_cache is not None
        ):
            sk_pts, _ = build_skeleton_segments(self._joints_cache)
            sk_cols = np.tile(
                np.array(s["smpl_mesh_color"], dtype=np.float32), (sk_pts.shape[0], 1)
            )
            self.lines_skel_r.update(sk_pts, sk_cols)
        else:
            self.lines_skel_r.count = 0

        if (
            s["show_mhr_skeleton"]
            and self.mhr is not None
            and self._mhr_joints_cache is not None
        ):
            sk_pts, sk_cols = build_parented_skeleton_segments(
                self._mhr_joints_cache,
                self.mhr.skeleton_parents,
                s["mhr_mesh_color"],
            )
            self.lines_mhr_skel_r.update(sk_pts, sk_cols)
        else:
            self.lines_mhr_skel_r.count = 0

        if s["show_xsens_skeleton"] and synced.xsens_segment_positions is not None:
            xsens_pts, xsens_cols = build_xsens_segments(
                synced.xsens_segment_positions[frame_idx], (1.0, 0.62, 0.12)
            )
            self.lines_xsens_r.update(xsens_pts, xsens_cols)
        else:
            self.lines_xsens_r.count = 0

        # Frustums (head/lwrist/rwrist/observer at current pose)
        if s["show_frustums"]:
            frust_local = _frustum_vertices_local(scale=0.10, fov=1.0, aspect=1.0)
            for tag, attr in (
                ("head", "T_world_head"),
                ("lwrist", "T_world_lwrist"),
                ("rwrist", "T_world_rwrist"),
                ("observer", "T_world_observer"),
            ):
                arr = getattr(synced, attr, None)
                if arr is None:
                    continue
                T = arr[frame_idx]
                world_pts = transform_points(T, frust_local)
                cols = np.tile(
                    np.array(self.TRAJ_COLORS[tag], dtype=np.float32),
                    (world_pts.shape[0], 1),
                )
                scene_pts.append(world_pts)
                scene_cols.append(cols)

        if scene_pts:
            all_pts = np.concatenate(scene_pts, axis=0)
            all_cols = np.concatenate(scene_cols, axis=0)
            self.lines_scene_r.update(all_pts, all_cols)
        else:
            self.lines_scene_r.count = 0

    # --------------------------------------------------------------- input

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self._active_splitter is not None:
            controls_w, rgb_w, _scene_x, _scene_w = self._layout_columns(
                self.wnd.buffer_size[0]
            )
            if self._active_splitter == "controls":
                controls_w += int(round(dx))
            elif self._active_splitter == "rgb":
                rgb_w += int(round(dx))
            self._set_layout_columns(controls_w, rgb_w)
            return
        if self.imgui.io.want_capture_mouse:
            self.imgui.mouse_drag_event(x, y, dx, dy)
            return
        # vigl Handler3D mapping (Pangolin conventions):
        #   left+right -> roll
        #   left       -> pan
        #   right      -> orbit
        #   middle     -> rotate-in-place
        states = self.wnd.mouse_states
        if states.left and states.right:
            self.camera.roll(dx, dy)
        elif states.left:
            self.camera.pan(dx, dy)
        elif states.right:
            self.camera.orbit(dx, dy)
        elif states.middle:
            self.camera.rotate_in_place(dx, dy)

    def on_mouse_position_event(self, x, y, dx, dy):
        if self._active_splitter is not None or self._hit_splitter(x) is not None:
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        self.imgui.mouse_position_event(x, y, dx, dy)

    def on_mouse_press_event(self, x, y, button):
        hit = self._hit_splitter(x)
        if hit is not None:
            self._active_splitter = hit
            controls_w, rgb_w, _scene_x, _scene_w = self._layout_columns(
                self.wnd.buffer_size[0]
            )
            self._set_layout_columns(controls_w, rgb_w)
            return
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        if self._active_splitter is not None:
            self._active_splitter = None
            return
        self.imgui.mouse_release_event(x, y, button)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if self.imgui.io.want_capture_mouse:
            self.imgui.mouse_scroll_event(x_offset, y_offset)
            return
        # vigl: positive delta zooms out, negative zooms in.
        # moderngl-window scroll: y_offset > 0 = wheel up = expect zoom in.
        self.camera.zoom(-float(y_offset))

    def on_key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.SPACE:
            self.state["playing"] = not self.state["playing"]
        elif key == self.wnd.keys.B:
            self.state["show_bbox"] = not self.state["show_bbox"]
        elif key == self.wnd.keys.F:
            self.state["follow"] = not self.state["follow"]
        elif key == self.wnd.keys.I:
            self.state["show_rgb"] = not self.state["show_rgb"]
        elif key == self.wnd.keys.L:
            self.state["show_bbox_labels"] = not self.state["show_bbox_labels"]
        elif key == self.wnd.keys.O:
            self.state["show_object_meshes"] = not self.state["show_object_meshes"]
        elif key == self.wnd.keys.S:
            show_smpl = not self.state["show_smpl_mesh"]
            self.state["show_smpl_mesh"] = show_smpl
            self.state["show_smpl_skeleton"] = show_smpl
        elif key == self.wnd.keys.T:
            self.state["show_traj"] = not self.state["show_traj"]
        elif key == self.wnd.keys.W:
            self.state["wireframe"] = not self.state["wireframe"]
        elif key == self.wnd.keys.X:
            self.state["show_xsens_skeleton"] = not self.state["show_xsens_skeleton"]

    def on_unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)

    def on_resize(self, width, height):
        self.imgui.resize(width, height)

    # --------------------------------------------------------------- frame

    def _step_smpl(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """SMPL forward + return (verts, joints)."""
        if self.smpl is None or self.synced.smpl_body_pose is None:
            return None
        import torch

        s = self.synced
        sl = slice(frame_idx, frame_idx + 1)
        self.smpl._ensure_model()
        with torch.no_grad():
            out = self.smpl._model(
                betas=torch.as_tensor(s.smpl_betas[sl], dtype=torch.float32),
                body_pose=torch.as_tensor(s.smpl_body_pose[sl], dtype=torch.float32),
                global_orient=torch.as_tensor(
                    s.smpl_global_orient[sl], dtype=torch.float32
                ),
                transl=torch.as_tensor(s.smpl_transl[sl], dtype=torch.float32),
            )
        verts = out.vertices.squeeze(0).cpu().numpy().astype(np.float32)
        joints = out.joints.squeeze(0).cpu().numpy().astype(np.float32)[:24]
        return verts, joints

    def _step_mhr(self, frame_idx: int) -> np.ndarray | None:
        """MHR skinning + return posed vertices."""
        if self.mhr is None:
            return None
        frame_idx = self._mhr_source_frame_idx(frame_idx)
        return self.mhr.forward(frame_idx)

    def _mhr_source_frame_idx(self, frame_idx: int) -> int:
        if self.synced.mhr_frame_indices is not None:
            frame_idx = int(self.synced.mhr_frame_indices[frame_idx])
        return int(frame_idx)

    _joints_cache: np.ndarray | None = None
    _mhr_joints_cache: np.ndarray | None = None

    def on_render(self, time, frametime):  # noqa: C901
        s = self.state
        synced = self.synced

        # Lazy point cloud load
        if not self._pcd_loaded and s["show_pcd"]:
            res = self._load_pcd_from_loader()
            if res is not None:
                pts, cols = res
                self.pcd_r.update(pts, cols)
                logger.info(f"loaded {pts.shape[0]} pointcloud points")
            self._pcd_loaded = True

        # Re-filter pcd if params changed
        if s["_pcd_dirty"]:
            res = self._load_pcd_from_loader()
            if res is not None:
                pts, cols = res
                self.pcd_r.update(pts, cols)
                logger.info(f"refiltered {pts.shape[0]} pointcloud points")
            s["_pcd_dirty"] = False

        # Advance playback
        if s["playing"]:
            s["frame"] = (s["frame"] + s["play_stride"]) % self.n_frames

        idx = int(s["frame"])

        # Body skinning/forward pass (only if frame changed or first call).
        if idx != self._last_frame_idx:
            if self.smpl is not None:
                sj = self._step_smpl(idx)
                if sj is not None and self.smpl_mesh_r is not None:
                    verts, joints = sj
                    normals = compute_normals(
                        verts, np.asarray(self.smpl.faces, dtype=np.int64)
                    )
                    self.smpl_mesh_r.update(verts, normals)
                    self._joints_cache = joints
            if self.mhr is not None:
                mhr_frame_idx = self._mhr_source_frame_idx(idx)
                if self.mhr_mesh_r is not None:
                    result = self.mhr.forward_with_joints(mhr_frame_idx)
                    verts, mhr_joints = result
                    normals = compute_normals(
                        verts, np.asarray(self.mhr.faces, dtype=np.int64)
                    )
                    self.mhr_mesh_r.update(verts, normals)
                    self._mhr_joints_cache = mhr_joints
                else:
                    self._mhr_joints_cache = self.mhr.skeleton_joints(mhr_frame_idx)
            self._build_dynamic_lines(idx)
            self._last_frame_idx = idx

        # Camera follow
        if s["follow"] and "head" in self._traj_positions:
            T = self.synced.T_world_head
            if T is not None:
                self.camera.follow(T[idx])
        else:
            # Reset prev when follow is off so re-enabling doesn't teleport.
            self.camera.reset_follow()

        # RGB
        if s["show_rgb"]:
            self._update_rgb(idx)

        # GL
        bg = float(s["bg_value"])
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.clear(bg, bg, bg, 1.0)
        w, h = self.wnd.buffer_size
        fb_w = max(w, 1)
        fb_h = max(h, 1)
        _controls_w, _rgb_w, scene_x, scene_w = self._layout_columns(fb_w)
        self.ctx.viewport = (scene_x, 0, scene_w, fb_h)
        self.ctx.scissor = (scene_x, 0, scene_w, fb_h)
        view = self.camera.view()
        proj = self.camera.proj(scene_w / fb_h)
        scene_viewport = (scene_w, fb_h)

        # Static scaffolding.
        if s["show_axes"]:
            self._maybe_rebuild_static()
            self.lines_static_r.draw(view, proj, scene_viewport)

        # Point cloud
        if s["show_pcd"]:
            self.pcd_r.draw(view, proj, point_size=s["pcd_point_size"])

        # Skeleton joints
        joint_pts = []
        joint_cols = []
        if s["show_smpl_skeleton"] and self._joints_cache is not None:
            jc = self._joints_cache
            joint_pts.append(jc.astype(np.float32))
            joint_cols.append(
                np.tile(
                    np.array(s["smpl_mesh_color"], dtype=np.float32), (jc.shape[0], 1)
                )
            )
        if s["show_mhr_skeleton"] and self._mhr_joints_cache is not None:
            jc = self._mhr_joints_cache
            joint_pts.append(jc.astype(np.float32))
            joint_cols.append(
                np.tile(
                    np.array(s["mhr_mesh_color"], dtype=np.float32), (jc.shape[0], 1)
                )
            )
        if s["show_xsens_skeleton"] and synced.xsens_segment_positions is not None:
            xsens_joints = synced.xsens_segment_positions[idx].astype(np.float32)
            joint_pts.append(xsens_joints)
            joint_cols.append(
                np.tile(
                    np.array((1.0, 0.62, 0.12), dtype=np.float32),
                    (xsens_joints.shape[0], 1),
                )
            )
        if joint_pts:
            self.joints_r.update(
                np.concatenate(joint_pts, axis=0), np.concatenate(joint_cols, axis=0)
            )
            self.joints_r.draw(view, proj, point_size=s["joint_size"])
        else:
            self.joints_r.count = 0

        # Lines (traj + frustums, then skeleton bones)
        if s["show_bbox"]:
            self.lines_bbox_r.draw(
                view, proj, scene_viewport, line_width=s["camera_line_width"]
            )
        self.lines_scene_r.draw(
            view, proj, scene_viewport, line_width=s["camera_line_width"]
        )
        self.lines_skel_r.draw(
            view, proj, scene_viewport, line_width=s["skeleton_line_width"]
        )
        self.lines_mhr_skel_r.draw(
            view, proj, scene_viewport, line_width=s["skeleton_line_width"]
        )
        self.lines_xsens_r.draw(
            view, proj, scene_viewport, line_width=s["skeleton_line_width"]
        )

        if s["show_object_meshes"]:
            self.object_mesh_r.draw(
                view,
                proj,
                s["wireframe"],
                alpha=float(s["object_mesh_alpha"]),
            )

        # Body meshes LAST so their alpha blends over the opaque scene behind them.
        if s["show_mhr_mesh"] and self.mhr_mesh_r is not None:
            self.mhr_mesh_r.draw(
                view,
                proj,
                s["mhr_mesh_color"],
                s["shading_mode"],
                s["wireframe"],
                alpha=float(s["mesh_alpha"]),
            )
        if s["show_smpl_mesh"] and self.smpl_mesh_r is not None:
            self.smpl_mesh_r.draw(
                view,
                proj,
                s["smpl_mesh_color"],
                s["shading_mode"],
                s["wireframe"],
                alpha=float(s["mesh_alpha"]),
            )

        # GUI
        self.ctx.scissor = None
        self.ctx.viewport = (0, 0, fb_w, fb_h)
        self._render_gui()
        self._render_bbox_labels(view, proj, scene_x, scene_w, fb_h)
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    # --------------------------------------------------------------- static

    def _layout_columns(self, width: int) -> tuple[int, int, int, int]:
        width = max(int(width), 1)
        controls_w = self._controls_w_user
        if controls_w is None:
            controls_w = min(330, max(260, int(width * 0.22)))
        rgb_w = self._rgb_w_user
        if rgb_w is None:
            rgb_w = 480
        if not self.state.get("show_rgb", True):
            rgb_w = 0
        return self._clamp_layout_columns(width, controls_w, rgb_w)

    def _clamp_layout_columns(
        self, width: int, controls_w: int, rgb_w: int
    ) -> tuple[int, int, int, int]:
        width = max(int(width), 1)
        min_scene_w = min(320, max(1, width // 4))
        side_budget = max(0, width - min_scene_w)
        controls_min = min(self.controls_min_w, side_budget)
        controls_max = min(self.controls_max_w, side_budget)
        controls_w = int(
            np.clip(controls_w, controls_min, max(controls_min, controls_max))
        )
        if not self.state.get("show_rgb", True):
            scene_x = controls_w
            scene_w = max(1, width - scene_x)
            return controls_w, 0, scene_x, scene_w
        rgb_budget = max(0, side_budget - controls_w)
        rgb_min = min(self.rgb_min_w, rgb_budget)
        rgb_max = min(self.rgb_max_w, rgb_budget)
        rgb_w = int(np.clip(rgb_w, rgb_min, max(rgb_min, rgb_max)))
        side_total = controls_w + rgb_w
        if side_total > side_budget:
            scale = side_budget / max(side_total, 1)
            controls_w = max(1, int(controls_w * scale))
            rgb_w = max(0, side_budget - controls_w)
        scene_x = controls_w + rgb_w
        scene_w = max(1, width - scene_x)
        return controls_w, rgb_w, scene_x, scene_w

    def _set_layout_columns(self, controls_w: int, rgb_w: int) -> None:
        width = max(self.wnd.buffer_size[0], 1)
        controls_w, rgb_w, _scene_x, _scene_w = self._clamp_layout_columns(
            width, controls_w, rgb_w
        )
        self._controls_w_user = controls_w
        self._rgb_w_user = rgb_w

    def _hit_splitter(self, x: float) -> str | None:
        width = max(self.wnd.buffer_size[0], 1)
        controls_w, rgb_w, _scene_x, _scene_w = self._layout_columns(width)
        hit = self.splitter_hit_px
        if abs(float(x) - float(controls_w)) <= hit:
            return "controls"
        if rgb_w > 0 and abs(float(x) - float(controls_w + rgb_w)) <= hit:
            return "rgb"
        return None

    _last_axes_show = None

    def _project_to_scene(
        self,
        point: np.ndarray,
        view: np.ndarray,
        proj: np.ndarray,
        scene_x: int,
        scene_w: int,
        height: int,
    ) -> tuple[float, float] | None:
        p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        clip = proj @ view @ p
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        x = float(scene_x) + (float(ndc[0]) * 0.5 + 0.5) * float(scene_w)
        y = (0.5 - float(ndc[1]) * 0.5) * float(height)
        if x < scene_x or x > scene_x + scene_w or y < 0 or y > height:
            return None
        return x, y

    def _render_bbox_labels(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        scene_x: int,
        scene_w: int,
        height: int,
    ) -> None:
        bbox = self.loader.bbox
        if (
            bbox is None
            or not self.state["show_bbox"]
            or not self.state["show_bbox_labels"]
        ):
            return

        draw_list = imgui.get_foreground_draw_list()
        text_col = imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        pad_x = 4.0
        pad_y = 2.0
        for center, label, color in zip(
            bbox.centers, bbox.labels, bbox.colors, strict=False
        ):
            screen = self._project_to_scene(
                center, view, proj, scene_x, scene_w, height
            )
            if screen is None:
                continue
            x, y = screen
            text_size = imgui.calc_text_size(label)
            pos = imgui.ImVec2(x + 5.0, y - text_size.y - 5.0)
            p_min = imgui.ImVec2(pos.x - pad_x, pos.y - pad_y)
            p_max = imgui.ImVec2(
                pos.x + text_size.x + pad_x, pos.y + text_size.y + pad_y
            )
            bg_col = imgui.get_color_u32(
                imgui.ImVec4(float(color[0]), float(color[1]), float(color[2]), 0.82)
            )
            draw_list.add_rect_filled(p_min, p_max, bg_col, 3.0)
            draw_list.add_text(pos, text_col, label)

    def _maybe_rebuild_static(self) -> None:
        s = self.state
        if s["show_axes"] == self._last_axes_show:
            return
        if s["show_axes"]:
            self.lines_static_r.update(self._static_pts, self._static_cols)
        else:
            self.lines_static_r.count = 0
        self._last_axes_show = s["show_axes"]

    # --------------------------------------------------------------- GUI

    def _render_gui(self) -> None:  # noqa: C901
        s = self.state
        imgui.new_frame()
        w, h = self.wnd.buffer_size
        controls_w, rgb_w, _scene_x, _scene_w = self._layout_columns(w)
        gui_h = max(h, 1)
        flags = (
            imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_saved_settings
        )

        imgui.set_next_window_pos((0, 0), imgui.Cond_.always)
        imgui.set_next_window_size((controls_w, gui_h), imgui.Cond_.always)
        imgui.begin("Controls", None, flags)

        imgui.text(f"Sequence: {self.synced.seq_name}")
        imgui.text(f"Frames: {self.n_frames}  @ {self.synced.fps:.1f} fps")
        imgui.separator()

        # Playback
        changed, v = imgui.slider_int(
            "Frame", int(s["frame"]), 0, max(self.n_frames - 1, 0)
        )
        if changed:
            s["frame"] = int(v)
        if imgui.button("Pause (Space)" if s["playing"] else "Play (Space)"):
            s["playing"] = not s["playing"]
        imgui.same_line()
        _, v = imgui.slider_int("Stride", int(s["play_stride"]), 1, 30)
        s["play_stride"] = int(v)
        _, s["follow"] = imgui.checkbox("Follow (f)", bool(s["follow"]))
        imgui.text(
            "Space=play  B=boxes  F=follow  I=RGB  L=labels  O=objects  S=SMPL  T=traj  W=wireframe  X=xsens"
        )
        imgui.text("L-drag=pan  R-drag=orbit  M-drag=look-around")
        imgui.text("L+R-drag=roll  scroll=zoom")
        imgui.separator()

        # Mesh
        if self.smpl_mesh_r is not None:
            _, s["show_smpl_mesh"] = imgui.checkbox(
                "SMPL mesh (s)", bool(s["show_smpl_mesh"])
            )
            imgui.same_line()
        if self.mhr_mesh_r is not None:
            _, s["show_mhr_mesh"] = imgui.checkbox("MHR mesh", bool(s["show_mhr_mesh"]))
            imgui.same_line()
        _, s["wireframe"] = imgui.checkbox("Wireframe (w)", bool(s["wireframe"]))
        if imgui.radio_button("Flat", s["shading_mode"] == 1):
            s["shading_mode"] = 1
        imgui.same_line()
        if imgui.radio_button("Normal", s["shading_mode"] == 0):
            s["shading_mode"] = 0
        if self.smpl_mesh_r is not None:
            _, c = imgui.color_edit3("SMPL color", list(s["smpl_mesh_color"]))
            s["smpl_mesh_color"] = (float(c[0]), float(c[1]), float(c[2]))
        if self.mhr_mesh_r is not None:
            _, c = imgui.color_edit3("MHR color", list(s["mhr_mesh_color"]))
            s["mhr_mesh_color"] = (float(c[0]), float(c[1]), float(c[2]))
        _, v = imgui.slider_float("Mesh alpha", float(s["mesh_alpha"]), 0.05, 1.0)
        s["mesh_alpha"] = float(v)
        _, v = imgui.slider_float("Background", float(s["bg_value"]), 0.0, 1.0)
        s["bg_value"] = float(v)

        # Skeleton
        if self.smpl is not None:
            _, s["show_smpl_skeleton"] = imgui.checkbox(
                "SMPL skeleton", bool(s["show_smpl_skeleton"])
            )
        if self.mhr is not None:
            _, s["show_mhr_skeleton"] = imgui.checkbox(
                "MHR skeleton", bool(s["show_mhr_skeleton"])
            )
        _, s["show_xsens_skeleton"] = imgui.checkbox(
            "XSens skeleton (x)", bool(s["show_xsens_skeleton"])
        )
        _, v = imgui.slider_float(
            "Skeleton line width", float(s["skeleton_line_width"]), 1.0, 10.0
        )
        s["skeleton_line_width"] = float(v)
        _, v = imgui.slider_float("Joint size", float(s["joint_size"]), 1.0, 20.0)
        s["joint_size"] = float(v)

        # Trajectories
        _, s["show_traj"] = imgui.checkbox("Trajectories (t)", bool(s["show_traj"]))
        _, v = imgui.slider_float(
            "Camera line width", float(s["camera_line_width"]), 1.0, 10.0
        )
        s["camera_line_width"] = float(v)
        _, v = imgui.slider_float("Trail sec", float(s["trail_seconds"]), 0.5, 30.0)
        s["trail_seconds"] = float(v)
        _, s["show_frustums"] = imgui.checkbox("Frustums", bool(s["show_frustums"]))
        if self.loader.bbox is not None:
            _, s["show_bbox"] = imgui.checkbox(
                "3D bounding boxes (b)", bool(s["show_bbox"])
            )
            _, s["show_bbox_labels"] = imgui.checkbox(
                "BBox labels (l)", bool(s["show_bbox_labels"])
            )
        if self.loader.mesh is not None:
            _, s["show_object_meshes"] = imgui.checkbox(
                "ShapeR meshes (o)", bool(s["show_object_meshes"])
            )
            _, v = imgui.slider_float(
                "ShapeR alpha", float(s["object_mesh_alpha"]), 0.05, 1.0
            )
            s["object_mesh_alpha"] = float(v)

        # Scaffolding
        _, s["show_axes"] = imgui.checkbox("Axes", bool(s["show_axes"]))

        # Point cloud
        imgui.separator()
        _, s["show_pcd"] = imgui.checkbox("Point cloud", bool(s["show_pcd"]))
        _, v = imgui.slider_float("PCD size", float(s["pcd_point_size"]), 1.0, 10.0)
        s["pcd_point_size"] = float(v)
        _, c = imgui.color_edit3("PCD color", list(s["pcd_color"]))
        new_col = (float(c[0]), float(c[1]), float(c[2]))
        if new_col != s["pcd_color"]:
            s["pcd_color"] = new_col
            if self.pcd_r.count > 0:
                cols = np.tile(
                    np.array(new_col, dtype=np.float32), (self.pcd_r.count, 1)
                )
                self.pcd_r.update_colors(cols)
        old = (s["pcd_dep"], s["pcd_invdep"])
        _, v = imgui.slider_float("Depth thr", float(s["pcd_dep"]), 0.001, 0.1, "%.3f")
        s["pcd_dep"] = float(v)
        _, v = imgui.slider_float(
            "InvDep thr", float(s["pcd_invdep"]), 0.0001, 0.005, "%.4f"
        )
        s["pcd_invdep"] = float(v)
        if (s["pcd_dep"], s["pcd_invdep"]) != old:
            s["_pcd_dirty"] = True

        # RGB
        imgui.separator()
        _, s["show_rgb"] = imgui.checkbox("RGB panel (i)", bool(s["show_rgb"]))
        if self.loader.bbox is not None and self.loader.bbox.bb2d_by_recording:
            changed, s["show_projected_bbox"] = imgui.checkbox(
                "Projected RGB boxes", bool(s["show_projected_bbox"])
            )
            labels_changed, s["show_projected_bbox_labels"] = imgui.checkbox(
                "Projected RGB labels", bool(s["show_projected_bbox_labels"])
            )
            if changed or labels_changed:
                self._invalidate_rgb_textures()

        imgui.end()

        # RGB panel
        if rgb_w > 0 and s["show_rgb"]:
            imgui.set_next_window_pos((controls_w, 0), imgui.Cond_.always)
            imgui.set_next_window_size((rgb_w, gui_h), imgui.Cond_.always)
            imgui.begin("RGB", None, flags)
            avail = imgui.get_content_region_avail()
            row_h = max((avail.y - 40.0) * 0.5, 1.0)
            self._render_rgb_texture("Head RGB", "head", row_h)
            imgui.separator()
            self._render_rgb_texture("Observer RGB", "observer", row_h)
            imgui.end()

    def _render_rgb_texture(self, label: str, tag: str, row_h: float) -> None:
        imgui.text(label)
        texture = self.rgb_textures[tag]
        if texture is None:
            return
        avail = imgui.get_content_region_avail()
        w, h = self.rgb_sizes[tag]
        if w <= 0 or h <= 0:
            return
        aspect = h / w
        disp_w = max(min(avail.x, row_h / aspect), 100.0)
        disp_h = disp_w * aspect
        imgui.image(texture.glo, (disp_w, disp_h))
        img_min = imgui.get_item_rect_min()
        self._render_projected_bbox_rgb_overlay(tag, img_min, disp_w, disp_h)

    def _render_projected_bbox_rgb_overlay(
        self,
        tag: str,
        img_min: Any,
        disp_w: float,
        disp_h: float,
    ) -> None:
        bbox = self.loader.bbox
        if bbox is None or not self.state["show_projected_bbox"]:
            return

        boxes = self.rgb_projected_boxes.get(tag, [])
        if not boxes:
            return

        tex_w, tex_h = self.rgb_sizes[tag]
        src_h = self.rgb_source_heights.get(tag, 0)
        if tex_w <= 0 or tex_h <= 0 or src_h <= 0:
            return

        draw_list = imgui.get_window_draw_list()
        scale_x = disp_w / float(tex_w)
        scale_y = disp_h / float(tex_h)

        for box in boxes:
            color = bbox.object_colors.get(box.object_uid, (0.0, 1.0, 0.0))
            col = imgui.get_color_u32(
                imgui.ImVec4(float(color[0]), float(color[1]), float(color[2]), 1.0)
            )
            for edge_pts, edge_valid in zip(box.points, box.valid, strict=False):
                for i in range(len(edge_pts) - 1):
                    if not (edge_valid[i] and edge_valid[i + 1]):
                        continue
                    x0_src, y0_src = edge_pts[i]
                    x1_src, y1_src = edge_pts[i + 1]
                    x0 = img_min.x + (float(src_h - 1) - float(y0_src)) * scale_x
                    y0 = img_min.y + float(x0_src) * scale_y
                    x1 = img_min.x + (float(src_h - 1) - float(y1_src)) * scale_x
                    y1 = img_min.y + float(x1_src) * scale_y
                    draw_list.add_line(
                        imgui.ImVec2(x0, y0),
                        imgui.ImVec2(x1, y1),
                        col,
                        _PROJECTED_BBOX_LINE_WIDTH,
                    )

        if not self.state["show_projected_bbox_labels"]:
            return

        imgui.set_window_font_scale(_PROJECTED_BBOX_LABEL_SCALE)
        try:
            for box in boxes:
                if box.label_xy is None:
                    continue
                color = bbox.object_colors.get(box.object_uid, (0.0, 1.0, 0.0))
                col = imgui.get_color_u32(
                    imgui.ImVec4(float(color[0]), float(color[1]), float(color[2]), 1.0)
                )
                label = bbox.object_labels.get(box.object_uid, str(box.object_uid))
                x_src, y_src = box.label_xy
                label_x = img_min.x + (float(src_h - 1) - float(y_src)) * scale_x
                label_y = img_min.y + float(x_src) * scale_y
                text_size = imgui.calc_text_size(label)
                x = float(
                    np.clip(
                        label_x - text_size.x * 0.5,
                        img_min.x,
                        img_min.x + disp_w - text_size.x,
                    )
                )
                y = float(
                    np.clip(
                        label_y - text_size.y * 0.5,
                        img_min.y,
                        img_min.y + disp_h - text_size.y,
                    )
                )
                draw_list.add_text(imgui.ImVec2(x, y), col, label)
        finally:
            imgui.set_window_font_scale(1.0)

    def _invalidate_rgb_textures(self) -> None:
        for tag in self._last_rgb_idx:
            self._last_rgb_idx[tag] = -1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch(
    loader: NymeriaPlusDataLoader,
    synced: SynchronizedSequence,
    smpl: SMPLBodyLoader | None,
    mhr: MHRBodyLoader | None = None,
    ui_scale: float | None = None,
) -> None:
    """Open a window and run the viewer until closed."""
    NymeriaPlusViewer._loader = loader
    NymeriaPlusViewer._synced = synced
    NymeriaPlusViewer._smpl = smpl
    NymeriaPlusViewer._mhr = mhr
    NymeriaPlusViewer._ui_scale = ui_scale
    mglw.run_window_config(NymeriaPlusViewer, args=["--window", "pyglet"])
