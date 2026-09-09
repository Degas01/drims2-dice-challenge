"""Full 6-DoF pose of the die from its top face, by PnP.

The homography mapper in :mod:`dice_vision.board_geometry` answers "where on the
board is the die", which is what the hands-on asks for.  This module answers the
stronger question -- "where is the die in space, and how is it turned" -- using
the pinhole model of the vision lecture rather than a plane-to-plane fit.

The idea is small: the top face of the die is a **square of known size**.  Four
image corners plus four object corners is exactly the input to Perspective-n-Point,
so the camera-to-die transform drops out of ``solvePnP``.  OpenCV's
``SOLVEPNP_IPPE_SQUARE`` is the specialised planar-square solver and is both
faster and better conditioned here than the general iterative one.

Why bother, when the homography already works:

* **No assumed height.**  The homography needs the die's size to be told to it
  in order to correct parallax; PnP measures the distance instead, from how big
  the square appears.
* **Yaw comes out directly**, in the camera frame, rather than being recovered
  by mapping two pixels through a plane fit.
* **It degrades honestly.**  The reprojection error is a real quality signal: a
  bad corner fit shows up as a large residual, so a wrong answer can be rejected
  instead of being returned with confidence.

The residual ambiguity is inherent, not a defect: a square viewed near
head-on has two nearly-equally-good poses (the classic planar flip), and it is
90-degree symmetric, so the yaw is only ever defined modulo 90 degrees.  Both
are handled explicitly below.

A caveat that measurement forced, and the important part of this module
--------------------------------------------------------------------------
PnP infers **depth from apparent size**: the square looks smaller, so it must be
further away.  That inference is only as good as the measured size, and here the
size is the weak link.  The die is about 25 px across and its quad comes from a
morphologically processed blob, so it runs systematically about 9% too large,
which lands directly as a 9% depth error.  Measured over 80 random scenes
against the renderer's exact ground truth:

============================================  ==========  ========
position estimator                            median      p95
============================================  ==========  ========
PnP, straight from the detector's quad         14.1 mm     47.6 mm
PnP, refined onto the known board plane         3.4 mm      6.4 mm
homography onto the known board plane           1.5 mm      3.0 mm
============================================  ==========  ========

(measured depth bias: -9.3% median.)

This is the vision lecture's "distances are not perceivable any more" made
concrete: one camera cannot measure range without a size prior, and this size
prior is poor.  The fix is not a better solver but more information -- the board
plane, whose height is known.  :func:`refine_pose_onto_plane` keeps PnP's
bearing and orientation, which are well conditioned, and replaces its range with
the ray-plane intersection; that alone recovers a factor of four.

The homography still wins on position, because it depends only on the blob's
centroid -- an average over hundreds of pixels, and so far less sensitive to the
segmentation bias than the quad's corners are.  Hence the division of labour:

* **position** from the plane constraint (the homography by default),
* **orientation, surface normal, and a reprojection residual that gates both**
  from PnP, whose yaw is good to 1.5 degrees median.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from dice_vision.camera_model import CameraModel

__all__ = [
    "SquarePose",
    "estimate_top_face_pose",
    "refine_pose_onto_plane",
    "die_centre_from_top_face",
    "pose_in_frame",
]


@dataclass
class SquarePose:
    """Pose of a planar square in the camera frame."""

    rotation: np.ndarray  # 3x3, square frame -> camera frame
    translation: np.ndarray  # 3-vector, square centre in the camera frame
    reprojection_error: float  # RMS, in pixels
    alternative: Optional["SquarePose"] = None  # the planar-flip twin, if any

    @property
    def distance(self) -> float:
        """Distance from the camera to the square's centre."""
        return float(np.linalg.norm(self.translation))

    @property
    def normal(self) -> np.ndarray:
        """Outward normal of the square, in the camera frame."""
        return self.rotation[:, 2]

    def yaw_about_normal(self) -> float:
        """In-plane rotation, folded into ``[-45, 45)`` degrees in radians.

        A square is unchanged by a quarter turn, so this is all the yaw that is
        observable from the shape alone.
        """
        angle = float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))
        return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _square_object_points(edge: float) -> np.ndarray:
    """Corners of a square of side ``edge``, centred at the origin, in its plane.

    Ordered to match :func:`dice_vision.detector._order_quad`: top-left,
    top-right, bottom-right, bottom-left as seen in the image, which for
    ``SOLVEPNP_IPPE_SQUARE`` must be counter-clockwise in the object plane with
    the normal towards the camera.
    """
    half = 0.5 * float(edge)
    return np.array(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _rms_reprojection(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: CameraModel,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera.matrix, camera.distortion.reshape(1, -1)
    )
    residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def estimate_top_face_pose(
    corners_px: Sequence[Sequence[float]],
    edge_length: float,
    camera: CameraModel,
    max_reprojection_error: float = 3.0,
) -> Optional[SquarePose]:
    """Recover the pose of the die's top face from its four image corners.

    Parameters
    ----------
    corners_px:
        The four corners, in the order :class:`~dice_vision.detector.Detection`
        reports them.
    edge_length:
        Physical edge length of the die, in metres.
    camera:
        Intrinsics and distortion.  Distortion is handled inside ``solvePnP``,
        so pass the corners as measured on the *raw* image.
    max_reprojection_error:
        Reject the solution above this RMS residual, in pixels.

    Returns
    -------
    The best pose, with the planar-flip twin attached as ``alternative``, or
    ``None`` if the geometry is degenerate or the fit is poor.
    """
    image_points = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if image_points.shape[0] != 4 or edge_length <= 0:
        return None

    # A collapsed quad (the die seen edge-on, or a bad fit) has no usable pose.
    area = float(cv2.contourArea(image_points.astype(np.float32)))
    if area < 16.0:
        return None

    object_points = _square_object_points(edge_length)
    distortion = camera.distortion.reshape(1, -1)

    # Try both windings.  A square's four corners can be listed clockwise or
    # counter-clockwise depending on how the caller found them, and pairing them
    # with the object points in the wrong sense asks the solver for a *mirrored*
    # correspondence, which no rigid pose can satisfy: it silently returns a
    # pose with a huge residual rather than an error.  Reversing is cheap, and
    # it makes the function work with whatever ordering the detector produces
    # instead of depending on a convention agreeing at a distance.
    #
    # Cyclic rotations need no such treatment: rotating the correspondence by
    # one corner is a quarter turn of the square about its own normal, which is
    # a genuine pose, and the yaw is only defined modulo 90 degrees anyway.
    solutions = []
    for candidate in (image_points, image_points[::-1]):
        try:
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points,
                np.ascontiguousarray(candidate),
                camera.matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            continue
        if not count:
            continue

        for rvec, tvec in zip(rvecs, tvecs):
            translation = np.asarray(tvec, dtype=float).reshape(3)
            if translation[2] <= 0:
                continue  # behind the camera
            rotation, _ = cv2.Rodrigues(rvec)
            solutions.append(
                SquarePose(
                    rotation=rotation,
                    translation=translation,
                    reprojection_error=_rms_reprojection(
                        object_points, candidate, rvec, tvec, camera
                    ),
                )
            )

    if not solutions:
        return None

    solutions.sort(key=lambda pose: pose.reprojection_error)
    best = solutions[0]
    if best.reprojection_error > max_reprojection_error:
        return None
    if len(solutions) > 1:
        best.alternative = solutions[1]
    return best


def refine_pose_onto_plane(
    pose: SquarePose,
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
) -> SquarePose:
    """Slide a PnP pose along its viewing ray until it sits on a known plane.

    PnP's *bearing* (the direction from the camera to the square) is set by the
    quad's centroid and is well conditioned; its *range* is set by the quad's
    size and is not.  Given the plane the square is known to lie on -- the board
    surface, one die-edge up -- the range follows from geometry instead of from
    apparent size, and the dominant error source disappears.

    Both the plane and the pose are in the camera frame.
    """
    point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)

    denominator = float(np.dot(normal, pose.translation))
    if abs(denominator) < 1e-9:
        return pose  # ray parallel to the plane: nothing sensible to do

    scale = float(np.dot(normal, point)) / denominator
    if scale <= 0:
        return pose

    return SquarePose(
        rotation=pose.rotation,
        translation=scale * pose.translation,
        reprojection_error=pose.reprojection_error,
        alternative=pose.alternative,
    )


def die_centre_from_top_face(pose: SquarePose, edge_length: float) -> np.ndarray:
    """Centre of the die, given the pose of its top face.

    The top face sits half an edge above the centre, along the face's own
    outward normal.  Reporting the centre rather than the face is what makes the
    number directly usable as a grasp target.
    """
    normal = pose.normal
    # The die's body lies *behind* its top face as seen from the camera, so the
    # centre is half an edge further away along the face normal. `translation`
    # points from the camera to the face, so "further away" is the direction
    # with a positive dot product against it.
    if float(np.dot(normal, pose.translation)) < 0:
        normal = -normal
    return pose.translation + 0.5 * float(edge_length) * normal


def pose_in_frame(
    pose_camera: np.ndarray,
    rotation_target_from_camera: np.ndarray,
    translation_target_from_camera: Sequence[float],
) -> np.ndarray:
    """Move a point from the camera frame into another frame."""
    rotation = np.asarray(rotation_target_from_camera, dtype=float).reshape(3, 3)
    translation = np.asarray(translation_target_from_camera, dtype=float).reshape(3)
    return rotation @ np.asarray(pose_camera, dtype=float).reshape(3) + translation
