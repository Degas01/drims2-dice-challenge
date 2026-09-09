"""Turning a pixel into a metric position on the DRIMS board.

Two mappers are provided, both ROS-free and unit-tested:

``HomographyMapper``
    Needs only the four board corners in the image and the board's physical
    size.  The detector can find those corners automatically
    (:meth:`~dice_vision.detector.DiceDetector.board_quad`), so this mapper
    calibrates itself from the first frame.  This is the pragmatic option the
    hands-on slides point at ("the green board is roughly 700mm x 500mm, (0,0)
    is roughly at the centre").

``PinholeMapper``
    Uses the real intrinsics from ``/oak/rgb/camera_info`` plus the extrinsic
    board pose.  More accurate when the camera is not close to nadir, and the
    right thing to use once the camera is properly hand-eye calibrated.

Parallax
--------
Both mappers take a ``height`` argument.  This matters more than it looks: the
camera sees the *top* face of the die, which floats one die-edge above the
board, so back-projecting that pixel onto the board plane puts the die too far
from the image centre.  For a 27 mm die on a 1 m high camera that is a ~2.7%
radial error -- around 9 mm at the edge of a 700 mm board, comfortably enough to
miss a grasp.  Passing ``height=dice_size`` removes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["HomographyMapper", "PinholeMapper", "DEFAULT_BOARD_SIZE_M"]

#: Board dimensions quoted in the DRIMS vision hands-on (metres).
DEFAULT_BOARD_SIZE_M: Tuple[float, float] = (0.70, 0.50)


@dataclass
class HomographyMapper:
    """Image -> board plane via a plane-to-plane homography.

    Parameters
    ----------
    homography:
        3x3 matrix mapping homogeneous image pixels to board coordinates on the
        plane ``z = 0``, in metres, origin at the board centre.
    camera_height_m:
        Height of the camera above the board.  Required only for the parallax
        correction; leave ``None`` to skip it.
    nadir_px:
        Pixel the camera looks straight down through (its principal point).
        Also only needed for the parallax correction.
    """

    homography: np.ndarray
    camera_height_m: Optional[float] = None
    nadir_px: Optional[Tuple[float, float]] = None

    @classmethod
    def from_board_quad(
        cls,
        quad: Sequence[Sequence[float]],
        board_size_m: Tuple[float, float] = DEFAULT_BOARD_SIZE_M,
        camera_height_m: Optional[float] = None,
        nadir_px: Optional[Tuple[float, float]] = None,
    ) -> "HomographyMapper":
        """Build from the four board corners, ordered TL, TR, BR, BL.

        The board's long side maps to X and the short side to Y, matching the
        robot base convention used by ``drims_description``.
        """
        half_x, half_y = 0.5 * board_size_m[0], 0.5 * board_size_m[1]
        src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        dst = np.array(
            [[-half_x, +half_y], [+half_x, +half_y], [+half_x, -half_y], [-half_x, -half_y]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src, dst)
        return cls(homography, camera_height_m, nadir_px)

    def pixel_to_board(
        self, uv: Sequence[float], height: float = 0.0
    ) -> Tuple[float, float, float]:
        """Map a pixel to a point ``height`` metres above the board plane."""
        point = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
        on_plane = cv2.perspectiveTransform(point, self.homography).reshape(2)
        x, y = float(on_plane[0]), float(on_plane[1])

        if height and self.camera_height_m and self.nadir_px is not None:
            nadir = cv2.perspectiveTransform(
                np.array([[list(self.nadir_px)]], dtype=np.float32), self.homography
            ).reshape(2)
            scale = (self.camera_height_m - height) / self.camera_height_m
            x = float(nadir[0] + (x - nadir[0]) * scale)
            y = float(nadir[1] + (y - nadir[1]) * scale)

        return x, y, float(height)

    def pixel_scale_at(self, uv: Sequence[float], height: float = 0.0) -> float:
        """Metres per pixel near ``uv`` -- used to convert the die's pixel size."""
        x0, y0, _ = self.pixel_to_board(uv, height)
        x1, y1, _ = self.pixel_to_board((uv[0] + 1.0, uv[1]), height)
        x2, y2, _ = self.pixel_to_board((uv[0], uv[1] + 1.0), height)
        return 0.5 * (float(np.hypot(x1 - x0, y1 - y0)) + float(np.hypot(x2 - x0, y2 - y0)))


@dataclass
class PinholeMapper:
    """Image -> board plane using intrinsics and the board's pose in the camera.

    ``rotation_board_from_camera`` and ``translation_board_from_camera`` describe
    the transform that takes a point expressed in the *camera* frame into the
    *board* frame, i.e. ``p_board = R @ p_cam + t``.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rotation_board_from_camera: np.ndarray
    translation_board_from_camera: np.ndarray

    @classmethod
    def from_camera_info(
        cls,
        k: Sequence[float],
        d: Sequence[float],
        rotation_board_from_camera: Sequence[Sequence[float]],
        translation_board_from_camera: Sequence[float],
    ) -> "PinholeMapper":
        """Build straight from a ``sensor_msgs/CameraInfo`` k and d array."""
        return cls(
            np.asarray(k, dtype=float).reshape(3, 3),
            np.asarray(d, dtype=float).reshape(1, -1),
            np.asarray(rotation_board_from_camera, dtype=float).reshape(3, 3),
            np.asarray(translation_board_from_camera, dtype=float).reshape(3),
        )

    def pixel_to_board(
        self, uv: Sequence[float], height: float = 0.0
    ) -> Tuple[float, float, float]:
        """Intersect the viewing ray with the plane ``z = height`` in board coords."""
        point = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float64)
        normalised = cv2.undistortPoints(point, self.camera_matrix, self.dist_coeffs)
        ray_cam = np.array([normalised[0, 0, 0], normalised[0, 0, 1], 1.0])

        origin = self.translation_board_from_camera
        direction = self.rotation_board_from_camera @ ray_cam

        if abs(direction[2]) < 1e-9:
            raise ValueError("viewing ray is parallel to the board plane")

        t = (height - origin[2]) / direction[2]
        if t <= 0:
            raise ValueError("board plane is behind the camera")

        hit = origin + t * direction
        return float(hit[0]), float(hit[1]), float(height)


def board_to_base(
    point_board: Sequence[float],
    board_origin_in_base: Sequence[float],
    board_yaw_in_base: float = 0.0,
) -> Tuple[float, float, float]:
    """Rigid transform from board coordinates to the robot ``base_link`` frame.

    The DRIMS boards sit flat in front of the arm, so a translation plus a yaw
    is enough; ``board_origin_in_base`` is where the board centre sits in the
    robot base frame (its Z is the ``surface_height`` the dice simulator uses).
    """
    c, s = np.cos(board_yaw_in_base), np.sin(board_yaw_in_base)
    x, y, z = (float(v) for v in point_board)
    return (
        float(board_origin_in_base[0] + c * x - s * y),
        float(board_origin_in_base[1] + s * x + c * y),
        float(board_origin_in_base[2] + z),
    )
