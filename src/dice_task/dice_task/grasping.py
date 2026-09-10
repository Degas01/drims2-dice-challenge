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

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "quaternion_from_matrix",
    "matrix_from_quaternion",
    "yaw_rotation",
    "normals_in_grasp_frame",
    "grasp_orientation",
    "tilt_for_turn",
    "flip_about_approach",
    "quaternion_distance",
    "nearest_equivalent_grasp",
    "approach_offset",
    "flange_position",
    "lowest_gripper_offset",
    "release_height",
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


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    c, s = np.cos(angle), np.sin(angle)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) * c + s * skew + (1 - c) * np.outer(k, k)


def grasp_orientation(
    grasp_axis: str, die_yaw: float = 0.0, close_axis: str = "y", tilt: float = 0.0
) -> np.ndarray:
    """Orientation of the tool for a grasp, as ``[x, y, z, w]``.

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
    tilt:
        Radians to lean the approach away from vertical, rotating about the
        closing axis.  Zero is straight down.  The fingers close along
        ``grasp_axis`` whatever the tilt, so they still land flat on the same
        two faces -- leaning only decides where the *body* of the gripper sits.

        That freedom is what makes a 90-degree flip work at all.  A flip rotates
        the tool by the same 90 degrees as the die, so a grasp that starts
        straight down ends up pointing sideways, with the gripper body swinging
        through the height of the die -- and the board.  Starting at -45 degrees
        ends at +45, so the gripper leans over the die at both ends and never
        goes near horizontal.  See :func:`tilt_for_turn`.
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
    if abs(tilt) > 1e-9:
        rot = _rotation_about(closing, tilt) @ rot
    return quaternion_from_matrix(rot)


def tilt_for_turn(quarter_turns: int, tilt: float) -> float:
    """The lean to start a turn with, so the finish leans the opposite way.

    A turn of ``+90`` degrees adds 90 to the approach angle, so starting at
    ``-tilt`` finishes at ``+tilt``: the excursion is centred on vertical
    instead of running from vertical to horizontal.
    """
    return -float(np.sign(quarter_turns)) * abs(tilt)


def flip_about_approach(quaternion: Sequence[float]) -> np.ndarray:
    """The same grasp with the gripper rolled 180 degrees about its own axis.

    The fingers swap sides, which grips exactly the same pair of faces from
    exactly the same direction -- so both orientations are equally valid, and
    the arm should be given whichever one its wrist is already nearer to.
    Handing it the wrong one costs a 180-degree spin of the last joint before
    the gripper has even touched the die.
    """
    rot = matrix_from_quaternion(quaternion)
    return quaternion_from_matrix(rot @ _rotation_about(np.array([0.0, 0.0, 1.0]), np.pi))


def quaternion_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle in radians between two orientations, ignoring quaternion sign."""
    dot = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def nearest_equivalent_grasp(
    quaternion: Sequence[float], reference: Optional[Sequence[float]]
) -> np.ndarray:
    """Pick between a grasp and its 180-degree roll, whichever is nearer.

    ``reference`` is the tool's current orientation.  With no reference the
    original is kept, so behaviour without TF is unchanged.
    """
    quaternion = np.asarray(quaternion, dtype=float)
    if reference is None:
        return quaternion
    flipped = flip_about_approach(quaternion)
    if quaternion_distance(flipped, reference) < quaternion_distance(quaternion, reference):
        return flipped
    return quaternion


def approach_offset(quaternion: Sequence[float], distance: float) -> np.ndarray:
    """Vector from a grasp pose back along the gripper's own approach axis.

    The textbook stand-off: it slides the fingers on and off the die along their
    own length rather than sweeping them sideways.

    **It is not what this project uses for the pre-grasp**, and the reason is
    worth keeping.  With a leaning grasp, backing off along the tool axis moves
    the pose sideways as well as up, and sideways is the expensive direction --
    see :func:`flange_position`.  On the DRIMS board it put the flange at
    0.847 m of a UR5e's 0.850 m reach and IK failed outright.  A vertical
    stand-off costs 0.789 m for the same grasp and is free of side effects,
    because the open fingers straddle the die along the grasp axis and so never
    touch it on a vertical descent however far the gripper leans.

    Kept because it is the right primitive for a gripper whose fingers *do* have
    to slide in past an obstruction, and because the measurement above is only
    meaningful next to the alternative.
    """
    approach = matrix_from_quaternion(quaternion)[:, 2]
    return -float(distance) * approach


def lowest_gripper_offset(
    quaternion: Sequence[float],
    body_radius: float = 0.045,
    body_offset: float = 0.05,
    body_length: float = 0.10,
) -> float:
    """How far below the tool tip the gripper's own body hangs, in metres.

    Model the gripper as a cylinder of radius ``body_radius`` running from
    ``body_offset`` to ``body_offset + body_length`` back along the approach
    axis, with the fingertips at the tip itself.  The answer is negative when
    some part of the gripper sits below the tip.

    This is what decides whether the die can be *placed* or has to be
    *released*.  Pointing straight down the body is directly above the tip and
    the answer is 0 -- the die can go all the way to the board.  Pointing
    horizontally, the body sweeps a full radius below the tip, and trying to
    lower the die to the board drives the gripper into it: MoveIt gets three
    quarters of the way down, stops, and reports 99999.
    """
    approach_z = float(np.asarray(matrix_from_quaternion(quaternion))[2, 2])
    # Vertical half-extent of the cylinder's cross-section.
    radius_drop = float(body_radius) * float(np.sqrt(max(0.0, 1.0 - approach_z**2)))
    near = -float(body_offset) * approach_z - radius_drop
    far = -float(body_offset + body_length) * approach_z - radius_drop
    return min(0.0, near, far)


def release_height(
    board_z: float,
    die_centre_z: float,
    quaternion: Sequence[float],
    clearance: float = 0.004,
    **gripper,
) -> float:
    """Lowest tool height that puts the die down without burying the gripper.

    Returns the height for the *tool tip*.  Where the gripper allows it this is
    the die's own resting height and the die is placed; where it does not, it is
    as low as the gripper can go, and the die falls the rest of the way.

    Dropping a die a few centimetres is not elegant, but it is what a parallel
    jaw gripper leaves you after a 90-degree flip, and it is self-correcting:
    the die lands flat on the face the turn selected, and if it does tumble the
    next loop reads the new face and re-plans.
    """
    resting = float(die_centre_z) + float(clearance)
    floor = float(board_z) + float(clearance) - lowest_gripper_offset(quaternion, **gripper)
    return max(resting, floor)


def flange_position(
    position: Sequence[float], quaternion: Sequence[float], tool_length: float
) -> np.ndarray:
    """Where the robot's flange sits when the tool tip is at ``position``.

    An arm's reach is quoted to its own wrist; everything past that is payload.
    So the question "can the robot get its fingers here" is really "can it get
    its *flange* to a point one gripper-length back along the approach axis",
    and a 150 mm gripper on a UR5e turns a comfortable 0.68 m grasp into a
    marginal 0.85 m one.

    It also means **orientation changes reachability at a fixed grasp point**,
    which is not obvious: leaning the tool 45 degrees swings the flange through
    several centimetres, and that is enough to cross the limit.
    """
    approach = matrix_from_quaternion(quaternion)[:, 2]
    return np.asarray(position, dtype=float) - float(tool_length) * approach


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
