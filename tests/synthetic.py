"""A small synthetic renderer for DRIMS-style board-and-die scenes.

It exists so the perception pipeline can be regression-tested without a robot,
a camera or a rosbag: every image comes with exact ground truth (face value,
board-frame position, yaw), and the nuisance factors the hands-on slides warn
about -- "each board has different light sources, because we live in a not ideal
world" -- are all parameterised.

The renderer is a real pinhole projection of a 3D scene (board plane at z=0,
die as a cube of edge ``size`` sitting on it), not a 2D sprite paste, so the
perspective of the die's top face and the parallax that the detector has to
correct for are both physically right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

BOARD_SIZE_M: Tuple[float, float] = (0.70, 0.50)

#: A palette that mirrors the dice visible in the DRIMS course photos, plus a
#: couple of deliberately awkward ones (white-on-white contrast, dark die).
DIE_PALETTE = {
    "yellow": ((40, 200, 235), (25, 25, 25)),
    "red": ((45, 45, 205), (240, 240, 240)),
    "blue": ((190, 90, 30), (245, 245, 245)),
    "green_dark": ((60, 110, 40), (245, 245, 245)),
    "orange": ((30, 130, 240), (20, 20, 20)),
    "black": ((35, 35, 38), (235, 235, 235)),
    "white": ((238, 238, 238), (30, 30, 30)),
    "purple": ((150, 60, 120), (240, 240, 240)),
}

PIP_LAYOUTS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
    6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
}


@dataclass
class SceneConfig:
    """Everything that can vary between two renders."""

    face_value: int = 5
    die_colour: str = "yellow"
    die_size_m: float = 0.027
    die_xy_m: Tuple[float, float] = (0.0, 0.0)
    die_yaw_rad: float = 0.0

    image_size: Tuple[int, int] = (1280, 720)
    focal_px: float = 900.0
    camera_height_m: float = 1.05
    camera_tilt_rad: Tuple[float, float] = (0.0, 0.0)  # about board X, Y
    camera_offset_m: Tuple[float, float] = (0.0, 0.0)

    board_bgr: Tuple[int, int, int] = (70, 175, 95)
    table_bgr: Tuple[int, int, int] = (58, 74, 96)

    light_gradient: float = 0.25  # 0 = flat lighting, 1 = extreme
    light_direction_rad: float = 0.9
    vignette: float = 0.22
    shadow_strength: float = 0.45
    shadow_offset_m: Tuple[float, float] = (0.02, 0.015)

    noise_sigma: float = 2.5
    jpeg_quality: Optional[int] = 88
    blur_px: int = 0

    clutter: bool = True
    seed: int = 0


@dataclass
class Scene:
    """A rendered image plus its ground truth."""

    image: np.ndarray
    config: SceneConfig
    die_center_px: Tuple[float, float]
    die_top_center_m: Tuple[float, float]
    board_quad_px: np.ndarray
    extra: dict = field(default_factory=dict)


def _camera(cfg: SceneConfig):
    """Return ``(K, rvec, tvec)`` for a camera looking down at the board."""
    w, h = cfg.image_size
    k = np.array(
        [[cfg.focal_px, 0.0, w / 2.0], [0.0, cfg.focal_px, h / 2.0], [0.0, 0.0, 1.0]]
    )

    # World: board plane z=0, X right, Y up (towards the far edge), Z up.
    # Camera looks down -Z, so its optical axis is -Z_world; the base rotation
    # flips Y so that image rows increase towards -Y_world.
    base = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tilt_x, tilt_y = cfg.camera_tilt_rad
    rx = cv2.Rodrigues(np.array([tilt_x, 0.0, 0.0]))[0]
    ry = cv2.Rodrigues(np.array([0.0, tilt_y, 0.0]))[0]
    rot_cam_from_world = rx @ ry @ base

    cam_pos = np.array(
        [cfg.camera_offset_m[0], cfg.camera_offset_m[1], cfg.camera_height_m]
    )
    tvec = -rot_cam_from_world @ cam_pos
    rvec = cv2.Rodrigues(rot_cam_from_world)[0]
    return k, rvec, tvec.reshape(3)


def _project(points_world, k, rvec, tvec):
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(pts, rvec, tvec, k, np.zeros(5))
    return projected.reshape(-1, 2)


def _die_corners(cfg: SceneConfig, z: float):
    half = cfg.die_size_m / 2.0
    c, s = np.cos(cfg.die_yaw_rad), np.sin(cfg.die_yaw_rad)
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    rotated = local @ np.array([[c, -s], [s, c]]).T
    cx, cy = cfg.die_xy_m
    return np.column_stack(
        (rotated[:, 0] + cx, rotated[:, 1] + cy, np.full(4, z))
    )


def render(cfg: SceneConfig) -> Scene:
    """Render one scene."""
    rng = np.random.default_rng(cfg.seed)
    w, h = cfg.image_size
    k, rvec, tvec = _camera(cfg)

    img = np.full((h, w, 3), cfg.table_bgr, np.uint8)

    # --- board ---
    half_x, half_y = BOARD_SIZE_M[0] / 2.0, BOARD_SIZE_M[1] / 2.0
    board_world = np.array(
        [[-half_x, half_y, 0.0], [half_x, half_y, 0.0],
         [half_x, -half_y, 0.0], [-half_x, -half_y, 0.0]]
    )
    board_px = _project(board_world, k, rvec, tvec)
    cv2.fillConvexPoly(img, board_px.astype(np.int32), cfg.board_bgr, cv2.LINE_AA)

    # Aluminium frame around the board, like the real cells.
    frame_world = board_world.copy()
    frame_world[:, 0] *= 1.05
    frame_world[:, 1] *= 1.07
    frame_px = _project(frame_world, k, rvec, tvec)
    cv2.polylines(img, [frame_px.astype(np.int32)], True, (170, 172, 175), 9, cv2.LINE_AA)

    if cfg.clutter:
        _add_clutter(img, rng, k, rvec, tvec)

    # --- cast shadow, on the board plane ---
    if cfg.shadow_strength > 0:
        shadow_world = _die_corners(cfg, 0.0)
        shadow_world[:, 0] += cfg.shadow_offset_m[0]
        shadow_world[:, 1] += cfg.shadow_offset_m[1]
        shadow_px = _project(shadow_world, k, rvec, tvec).astype(np.int32)
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, shadow_px, (0, 0, 0), cv2.LINE_AA)
        blurred = cv2.GaussianBlur(overlay, (31, 31), 0)
        alpha = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(alpha, shadow_px, cfg.shadow_strength)
        alpha = cv2.GaussianBlur(alpha, (41, 41), 0)[..., None]
        img = (img * (1 - alpha) + blurred * alpha).astype(np.uint8)

    # --- die ---
    body, pip_colour = DIE_PALETTE[cfg.die_colour]
    top_world = _die_corners(cfg, cfg.die_size_m)
    bottom_world = _die_corners(cfg, 0.0)
    top_px = _project(top_world, k, rvec, tvec)
    bottom_px = _project(bottom_world, k, rvec, tvec)

    # Draw the four vertical faces first (only the ones facing the camera show).
    shaded = tuple(int(v * 0.72) for v in body)
    for i in range(4):
        j = (i + 1) % 4
        side = np.array([top_px[i], top_px[j], bottom_px[j], bottom_px[i]])
        if cv2.contourArea(side.astype(np.float32)) > 1.0:
            cv2.fillConvexPoly(img, side.astype(np.int32), shaded, cv2.LINE_AA)

    cv2.fillConvexPoly(img, top_px.astype(np.int32), body, cv2.LINE_AA)

    # --- pips on the top face ---
    step = cfg.die_size_m * 0.26
    radius_m = cfg.die_size_m * 0.105
    c, s = np.cos(cfg.die_yaw_rad), np.sin(cfg.die_yaw_rad)
    pip_px = []
    for gx, gy in PIP_LAYOUTS[cfg.face_value]:
        lx, ly = gx * step, gy * step
        wx = cfg.die_xy_m[0] + c * lx - s * ly
        wy = cfg.die_xy_m[1] + s * lx + c * ly
        centre = _project([[wx, wy, cfg.die_size_m]], k, rvec, tvec)[0]
        edge = _project([[wx + radius_m, wy, cfg.die_size_m]], k, rvec, tvec)[0]
        r = max(2, int(round(np.linalg.norm(edge - centre))))
        cv2.circle(img, tuple(centre.astype(int)), r, pip_colour, -1, cv2.LINE_AA)
        pip_px.append(centre)

    # --- illumination, noise, compression ---
    img = _apply_lighting(img, cfg)
    if cfg.blur_px:
        ksize = int(cfg.blur_px) | 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    if cfg.noise_sigma > 0:
        noise = rng.normal(0.0, cfg.noise_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if cfg.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    die_center_px = top_px.mean(axis=0)
    return Scene(
        image=img,
        config=cfg,
        die_center_px=(float(die_center_px[0]), float(die_center_px[1])),
        die_top_center_m=(float(cfg.die_xy_m[0]), float(cfg.die_xy_m[1])),
        board_quad_px=board_px.astype(np.float32),
        extra={"pips_px": np.array(pip_px), "camera_matrix": k},
    )


def _apply_lighting(img: np.ndarray, cfg: SceneConfig) -> np.ndarray:
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs - w / 2.0) / (w / 2.0)
    ny = (ys - h / 2.0) / (h / 2.0)

    gain = np.ones((h, w), np.float32)
    if cfg.light_gradient:
        direction = np.array(
            [np.cos(cfg.light_direction_rad), np.sin(cfg.light_direction_rad)]
        )
        gain += cfg.light_gradient * (nx * direction[0] + ny * direction[1])
    if cfg.vignette:
        gain *= 1.0 - cfg.vignette * np.clip(nx**2 + ny**2, 0.0, 1.6) / 1.6

    return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def _add_clutter(img, rng, k, rvec, tvec) -> None:
    """Cables, clamps and a bit of the arm intruding at the board edge."""
    h, w = img.shape[:2]

    # A dark cable snaking in from the left, drawn on the board plane.
    xs = np.linspace(-0.36, -0.20, 24)
    ys = 0.16 + 0.03 * np.sin(np.linspace(0, 3.4, 24)) + rng.normal(0, 0.002, 24)
    cable = _project(np.column_stack((xs, ys, np.zeros_like(xs))), k, rvec, tvec)
    cv2.polylines(img, [cable.astype(np.int32)], False, (28, 28, 32), 7, cv2.LINE_AA)

    # Two clamps on the near frame edge.
    for cx in (-0.22, 0.24):
        corners = np.array(
            [[cx - 0.035, -0.27, 0.0], [cx + 0.035, -0.27, 0.0],
             [cx + 0.035, -0.235, 0.0], [cx - 0.035, -0.235, 0.0]]
        )
        pts = _project(corners, k, rvec, tvec).astype(np.int32)
        cv2.fillConvexPoly(img, pts, (110, 70, 40), cv2.LINE_AA)

    # Corner of the table showing past the frame.
    cv2.rectangle(img, (0, 0), (int(0.05 * w), h), (52, 66, 86), -1)


def random_scene(rng: np.random.Generator, **overrides) -> SceneConfig:
    """A randomised but plausible scene configuration."""
    cfg = SceneConfig(
        face_value=int(rng.integers(1, 7)),
        die_colour=str(rng.choice(list(DIE_PALETTE))),
        die_size_m=float(rng.uniform(0.024, 0.032)),
        die_xy_m=(float(rng.uniform(-0.26, 0.26)), float(rng.uniform(-0.17, 0.17))),
        die_yaw_rad=float(rng.uniform(-np.pi, np.pi)),
        camera_height_m=float(rng.uniform(0.85, 1.25)),
        camera_tilt_rad=(float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05))),
        camera_offset_m=(float(rng.uniform(-0.04, 0.04)), float(rng.uniform(-0.04, 0.04))),
        board_bgr=(
            int(rng.integers(55, 90)),
            int(rng.integers(150, 200)),
            int(rng.integers(75, 115)),
        ),
        light_gradient=float(rng.uniform(0.05, 0.38)),
        light_direction_rad=float(rng.uniform(0, 2 * np.pi)),
        vignette=float(rng.uniform(0.08, 0.30)),
        shadow_strength=float(rng.uniform(0.15, 0.55)),
        shadow_offset_m=(float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.03, 0.03))),
        noise_sigma=float(rng.uniform(1.0, 4.5)),
        jpeg_quality=int(rng.integers(78, 96)),
        seed=int(rng.integers(0, 1 << 30)),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
