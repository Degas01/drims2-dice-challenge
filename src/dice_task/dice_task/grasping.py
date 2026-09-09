"""Grasp and re-orientation pose maths, free of ROS so it can be unit-tested.

The die rests on the board with some yaw about the vertical, and a parallel-jaw
gripper coming straight down can only close on a pair of opposite *vertical*
faces.  Those faces are square-on to the die, not to the robot base, so the
grasp direction has to follow the die's yaw.

That is handled by planning in a **grasp frame**: the robot base frame rotated
about Z by the die's yaw.  In that frame the die's lateral faces are axis
aligned, the planner's ``"x"`` and ``"y"`` grasp axes mean exactly "close the
fingers along X / along Y", and everything the planner computes converts back to
the base frame with a single yaw rotation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "quaternion_from_matrix",
    "matrix_from_quaternion",
    "yaw_rotation",
    "normals_in_grasp_frame",
    "grasp_orientation",
    "rotate_about_axis",
    "AXIS_VECTORS",
]

AXIS_VECTORS: Dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


# --------------------------------------------------------------------------- #
# Quaternion helpers (x, y, z, w -- the ROS ordering)
# --------------------------------------------------------------------------- #


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion ``[x, y, z, w]``, using Shepperd's method."""
    m = np.asarray(matrix, dtype=float)[:3, :3]
    trace = float(np.trace(m))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([x, y, z, w], dtype=float)
    # Canonical sign, so that comparing two orientations is unambiguous.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion / np.linalg.norm(quaternion)


def matrix_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` -> rotation matrix."""
    x, y, z, w = (float(v) for v in quaternion)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("cannot normalise a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def axis_angle_matrix(axis: str, angle: float) -> np.ndarray:
    """Rotation of ``angle`` radians about a named world axis."""
    k = AXIS_VECTORS[axis]
    c, s = np.cos(angle), np.sin(angle)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) * c + s * skew + (1 - c) * np.outer(k, k)


def yaw_rotation(yaw: float) -> np.ndarray:
    """Rotation about the vertical axis."""
    return axis_angle_matrix("z", yaw)


# --------------------------------------------------------------------------- #
# Grasp frame
# --------------------------------------------------------------------------- #


def normals_in_grasp_frame(
    normals_in_base: Dict[int, Sequence[float]], die_yaw: float
) -> Dict[int, np.ndarray]:
    """Rotate measured face normals from the base frame into the grasp frame."""
    rot = yaw_rotation(-die_yaw)
    return {face: rot @ np.asarray(n, dtype=float) for face, n in normals_in_base.items()}


def grasp_orientation(
    grasp_axis: str, die_yaw: float = 0.0, close_axis: str = "y"
) -> np.ndarray:
    """Orientation of the tool for a top-down grasp, as ``[x, y, z, w]``.

    Parameters
    ----------
    grasp_axis:
        ``"x"`` or ``"y"``: the horizontal axis, *in the grasp frame*, that the
        fingers close along.  It is also the axis the wrist will rotate about,
        because the fingers must sit on the two faces the rotation leaves put.
    die_yaw:
        The die's yaw about the vertical, in the base frame.
    close_axis:
        Which axis of the *tool* frame the fingers close along.  Robotiq-style
        grippers close along the tool Y; set ``"x"`` for a tool that closes
        along X.
    """
    if grasp_axis not in ("x", "y"):
        raise ValueError(f"grasp_axis must be 'x' or 'y', got {grasp_axis!r}")
    if close_axis not in ("x", "y"):
        raise ValueError(f"close_axis must be 'x' or 'y', got {close_axis!r}")

    # Approach straight down: the tool's Z points at the board.
    tool_z = np.array([0.0, 0.0, -1.0])
    closing = yaw_rotation(die_yaw) @ AXIS_VECTORS[grasp_axis]

    if close_axis == "y":
        tool_y = closing
        tool_x = np.cross(tool_y, tool_z)
    else:
        tool_x = closing
        tool_y = np.cross(tool_z, tool_x)

    rot = np.column_stack((tool_x, tool_y, tool_z))
    return quaternion_from_matrix(rot)


def rotate_about_axis(
    position: Sequence[float],
    quaternion: Sequence[float],
    grasp_axis: str,
    quarter_turns: int,
    pivot: Sequence[float],
    die_yaw: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a held pose by 90 degrees about a horizontal axis through ``pivot``.

    Rotating about an axis through the *die's* centre rather than through the
    tool flange is what keeps the die roughly in place while it turns; rotating
    about the flange would swing it through a wide arc, which is how you clip
    the board (the "BE CAREFUL WITH ROTATIONS" slide in the DRIMS hands-on).
    """
    if quarter_turns not in (-2, -1, 1, 2):
        raise ValueError(f"quarter_turns must be +/-1 or +/-2, got {quarter_turns}")

    axis_world = yaw_rotation(die_yaw) @ AXIS_VECTORS[grasp_axis]
    angle = 0.5 * np.pi * quarter_turns
    k = axis_world / np.linalg.norm(axis_world)
    c, s = np.cos(angle), np.sin(angle)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) * c + s * skew + (1 - c) * np.outer(k, k)

    pivot = np.asarray(pivot, dtype=float)
    position = np.asarray(position, dtype=float)
    new_position = pivot + rot @ (position - pivot)
    new_matrix = rot @ matrix_from_quaternion(quaternion)
    return new_position, quaternion_from_matrix(new_matrix)
