"""The pinhole camera model, in the form the DRIMS vision lecture presents it.

Projection
----------
Central projection is linear in homogeneous coordinates::

    x = P X,      P = K [R | t]

with the intrinsic matrix carrying the focal length and the principal point
offset (the lecture derives the offset separately, because the image origin is a
corner and not the optical axis)::

        | fx   0  cx |
    K = |  0  fy  cy |
        |  0   0   1 |

so a point ``(X, Y, Z)`` in the camera frame lands at
``(fx X/Z + cx, fy Y/Z + cy)``.

Distortion
----------
Real lenses are the lecture's "non-ideal cameras".  The normalised coordinates
are displaced radially and tangentially before being scaled by ``K``::

    x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
    y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y

That model has no closed-form inverse, so :meth:`CameraModel.undistort_normalised`
inverts it by fixed-point iteration.  The hands-on slides are explicit that this
matters -- *"undistort the image to have better accuracy"* -- and at the edge of
a wide-angle frame it is worth several millimetres on the board.

Everything here is ROS-free; ``CameraModel.from_camera_info`` takes the ``k`` and
``d`` arrays of a ``sensor_msgs/CameraInfo`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["CameraModel"]


@dataclass
class CameraModel:
    """Intrinsics plus lens distortion for a single camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    #: ``[k1, k2, p1, p2, k3]`` -- OpenCV's ordering, and ``CameraInfo.d``'s.
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    image_size: Optional[Tuple[int, int]] = None  # (width, height)

    def __post_init__(self) -> None:
        self.distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if self.distortion.size < 5:
            self.distortion = np.pad(self.distortion, (0, 5 - self.distortion.size))
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_camera_info(
        cls,
        k: Sequence[float],
        d: Sequence[float] = (),
        image_size: Optional[Tuple[int, int]] = None,
    ) -> "CameraModel":
        """Build from a ``sensor_msgs/CameraInfo``'s ``k`` and ``d`` arrays."""
        k = np.asarray(k, dtype=float).reshape(3, 3)
        return cls(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            distortion=np.asarray(d, dtype=float).reshape(-1) if len(d) else np.zeros(5),
            image_size=image_size,
        )

    @property
    def matrix(self) -> np.ndarray:
        """The intrinsic matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def principal_point(self) -> Tuple[float, float]:
        """Where the optical axis pierces the image -- the camera's nadir."""
        return (self.cx, self.cy)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.distortion) > 1e-12))

    # ------------------------------------------------------------------ #
    # Forward: 3D -> pixels
    # ------------------------------------------------------------------ #

    def distort_normalised(self, points: np.ndarray) -> np.ndarray:
        """Apply the radial-tangential model to normalised coordinates."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        k1, k2, p1, p2, k3 = self.distortion[:5]
        x, y = points[:, 0], points[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.column_stack((x_d, y_d))

    def project(self, points_camera: np.ndarray) -> np.ndarray:
        """Project 3D points expressed in the camera frame to pixels."""
        points = np.asarray(points_camera, dtype=float).reshape(-1, 3)
        z = points[:, 2]
        if np.any(z <= 1e-9):
            raise ValueError("cannot project points at or behind the camera plane")
        normalised = points[:, :2] / z[:, None]
        distorted = self.distort_normalised(normalised)
        return np.column_stack(
            (self.fx * distorted[:, 0] + self.cx, self.fy * distorted[:, 1] + self.cy)
        )

    # ------------------------------------------------------------------ #
    # Inverse: pixels -> rays
    # ------------------------------------------------------------------ #

    def to_normalised(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels to *distorted* normalised coordinates (just undo ``K``)."""
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        return np.column_stack(
            ((pixels[:, 0] - self.cx) / self.fx, (pixels[:, 1] - self.cy) / self.fy)
        )

    def undistort_normalised(
        self, distorted: np.ndarray, iterations: int = 20, tolerance: float = 1e-12
    ) -> np.ndarray:
        """Invert the distortion model by fixed-point iteration.

        The forward model has no closed form inverse.  Starting from the
        distorted point and repeatedly re-applying the correction converges
        quickly for the distortion magnitudes real lenses have; the iteration
        stops early once it stops moving.
        """
        distorted = np.asarray(distorted, dtype=float).reshape(-1, 2)
        if not self.has_distortion:
            return distorted.copy()

        k1, k2, p1, p2, k3 = self.distortion[:5]
        undistorted = distorted.copy()
        for _ in range(iterations):
            x, y = undistorted[:, 0], undistorted[:, 1]
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            updated = np.column_stack(
                ((distorted[:, 0] - dx) / radial, (distorted[:, 1] - dy) / radial)
            )
            shift = float(np.max(np.abs(updated - undistorted)))
            undistorted = updated
            if shift < tolerance:
                break
        return undistorted

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Move pixels to where an ideal pinhole lens would have put them."""
        normalised = self.undistort_normalised(self.to_normalised(pixels))
        return np.column_stack(
            (self.fx * normalised[:, 0] + self.cx, self.fy * normalised[:, 1] + self.cy)
        )

    def ray(self, pixel: Sequence[float]) -> np.ndarray:
        """Unit viewing direction through a pixel, in the camera frame."""
        normalised = self.undistort_normalised(self.to_normalised([pixel]))[0]
        direction = np.array([normalised[0], normalised[1], 1.0])
        return direction / np.linalg.norm(direction)

    def backproject(self, pixel: Sequence[float], depth: float) -> np.ndarray:
        """The 3D point at ``depth`` (along +Z) under a pixel, in camera coords."""
        normalised = self.undistort_normalised(self.to_normalised([pixel]))[0]
        return np.array([normalised[0] * depth, normalised[1] * depth, float(depth)])

    # ------------------------------------------------------------------ #
    # Whole images
    # ------------------------------------------------------------------ #

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Rectify a whole frame.

        Keeps the same intrinsics for the output, so pixel coordinates in the
        rectified image can be used directly with this model's ``fx, fy, cx, cy``
        and zero distortion.
        """
        if not self.has_distortion:
            return image
        return cv2.undistort(image, self.matrix, self.distortion.reshape(1, -1))

    def without_distortion(self) -> "CameraModel":
        """The same camera, as if the lens were ideal."""
        return CameraModel(self.fx, self.fy, self.cx, self.cy, np.zeros(5), self.image_size)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def field_of_view(self) -> Tuple[float, float]:
        """Horizontal and vertical field of view in radians."""
        if self.image_size is None:
            raise ValueError("image_size is required to compute the field of view")
        width, height = self.image_size
        return (
            2.0 * float(np.arctan(0.5 * width / self.fx)),
            2.0 * float(np.arctan(0.5 * height / self.fy)),
        )

    def metres_per_pixel(self, depth: float) -> float:
        """Scale at a given depth -- how big a pixel is on the board."""
        return float(depth * 0.5 * (1.0 / self.fx + 1.0 / self.fy))
