"""Tests for the grasp/re-orientation pose maths (no ROS required)."""

import numpy as np
import pytest

from dice_task.die_model import FACE_NORMALS, Regrasp, apply, plan_from_normals, up_face
from dice_task.grasping import (
    AXIS_VECTORS,
    approach_offset,
    flip_about_approach,
    grasp_orientation,
    nearest_equivalent_grasp,
    matrix_from_quaternion,
    normals_in_grasp_frame,
    quaternion_distance,
    quaternion_from_matrix,
    rotate_about_axis,
    tilt_for_turn,
    yaw_rotation,
)

from test_die_model import all_orientations


# --------------------------------------------------------------------------- #
# Quaternion round trips
# --------------------------------------------------------------------------- #


def test_quaternion_round_trip_over_the_cube_group():
    for rot in all_orientations():
        q = quaternion_from_matrix(rot)
        assert np.allclose(matrix_from_quaternion(q), rot, atol=1e-9)


def test_quaternion_round_trip_over_random_rotations():
    rng = np.random.default_rng(4)
    for _ in range(500):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) * c + s * skew + (1 - c) * np.outer(axis, axis)
        assert np.allclose(matrix_from_quaternion(quaternion_from_matrix(rot)), rot, atol=1e-8)


def test_quaternions_are_unit_and_canonically_signed():
    for rot in all_orientations():
        q = quaternion_from_matrix(rot)
        assert pytest.approx(1.0, abs=1e-9) == float(np.linalg.norm(q))
        assert q[3] >= -1e-12


def test_matrix_from_quaternion_rejects_zero():
    with pytest.raises(ValueError):
        matrix_from_quaternion([0.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Grasp orientation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("grasp_axis", ["x", "y"])
@pytest.mark.parametrize("close_axis", ["x", "y"])
@pytest.mark.parametrize("die_yaw_deg", [0, 17, 45, 90, -33])
def test_grasp_orientation_points_the_tool_down(grasp_axis, close_axis, die_yaw_deg):
    q = grasp_orientation(grasp_axis, np.deg2rad(die_yaw_deg), close_axis)
    rot = matrix_from_quaternion(q)
    tool_z = rot[:, 2]
    assert np.allclose(tool_z, [0.0, 0.0, -1.0], atol=1e-9), "gripper must approach from above"


@pytest.mark.parametrize("grasp_axis", ["x", "y"])
@pytest.mark.parametrize("close_axis", ["x", "y"])
@pytest.mark.parametrize("die_yaw_deg", [0, 17, 45, -33])
def test_fingers_close_along_the_grasp_axis(grasp_axis, close_axis, die_yaw_deg):
    """The fingers must sit on the faces the rotation leaves in place."""
    yaw = np.deg2rad(die_yaw_deg)
    q = grasp_orientation(grasp_axis, yaw, close_axis)
    rot = matrix_from_quaternion(q)
    actual = rot[:, 0] if close_axis == "x" else rot[:, 1]
    expected = yaw_rotation(yaw) @ AXIS_VECTORS[grasp_axis]
    assert np.allclose(actual, expected, atol=1e-9)


def test_grasp_orientation_is_a_proper_rotation():
    for grasp_axis in ("x", "y"):
        for close_axis in ("x", "y"):
            rot = matrix_from_quaternion(grasp_orientation(grasp_axis, 0.3, close_axis))
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
            assert pytest.approx(1.0, abs=1e-9) == float(np.linalg.det(rot))


def test_grasp_orientation_rejects_bad_axes():
    with pytest.raises(ValueError):
        grasp_orientation("z")
    with pytest.raises(ValueError):
        grasp_orientation("x", close_axis="z")


# --------------------------------------------------------------------------- #
# Re-orientation
# --------------------------------------------------------------------------- #


def test_rotation_keeps_the_die_near_the_pivot():
    """Rotating about the die's own centre must not swing it across the board."""
    pivot = np.array([0.55, 0.10, 0.25])
    tool = pivot + np.array([0.0, 0.0, 0.12])  # flange 12 cm above the die
    for axis in ("x", "y"):
        for turns in (1, -1):
            moved, _ = rotate_about_axis(
                tool, grasp_orientation(axis), axis, turns, pivot
            )
            assert np.linalg.norm(moved - pivot) == pytest.approx(0.12, abs=1e-9)


def test_four_quarter_turns_restore_the_pose():
    pivot = np.array([0.5, 0.0, 0.2])
    position = pivot + np.array([0.03, -0.02, 0.10])
    quaternion = grasp_orientation("x")
    for axis in ("x", "y"):
        p, q = position, quaternion
        for _ in range(4):
            p, q = rotate_about_axis(p, q, axis, 1, pivot)
        assert np.allclose(p, position, atol=1e-9)
        assert np.allclose(q, quaternion, atol=1e-9)


def test_rotation_rejects_bad_turn_counts():
    with pytest.raises(ValueError):
        rotate_about_axis([0, 0, 0], [0, 0, 0, 1], "x", 3, [0, 0, 0])


# --------------------------------------------------------------------------- #
# The whole chain: measured normals -> plan -> executed rotation -> new face up
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("die_yaw_deg", [0, 23, 45, -61, 90])
def test_planned_rotations_actually_expose_the_target_face(die_yaw_deg):
    """End-to-end: the pose maths and the planner must agree about the axes.

    The planner works in the grasp frame; the executor rotates about the same
    axis expressed in the base frame.  If those two disagree by the die's yaw --
    an easy mistake, and one that silently turns the die the wrong way -- this
    test fails.
    """
    yaw = np.deg2rad(die_yaw_deg)
    yaw_matrix = yaw_rotation(yaw)

    for orientation in all_orientations():
        # The die really sits yawed in the base frame.
        base_orientation = yaw_matrix @ orientation
        normals_base = {f: base_orientation @ n for f, n in FACE_NORMALS.items()}

        for target in range(1, 7):
            grasp_normals = normals_in_grasp_frame(normals_base, yaw)
            plan = plan_from_normals(grasp_normals, target)

            # Execute the plan on the real (yawed) die.
            state = base_orientation
            for move in plan:
                axis_world = yaw_matrix @ AXIS_VECTORS[move.axis]
                angle = 0.5 * np.pi * move.quarter_turns
                k = axis_world
                c, s = np.cos(angle), np.sin(angle)
                skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                rot = np.eye(3) * c + s * skew + (1 - c) * np.outer(k, k)
                state = rot @ state

            achieved = max(
                range(1, 7), key=lambda f: float((state @ FACE_NORMALS[f])[2])
            )
            assert achieved == target, (
                f"yaw {die_yaw_deg} deg, target {target}: ended on {achieved}"
            )


def test_face_selection_is_insensitive_to_yaw_but_grasping_is_not():
    """Where the die's yaw does and does not matter.

    Choosing *which* face to bring up is yaw-robust: as long as the yaw is under
    45 degrees the nearest cube orientation is the un-yawed one, so the planner
    picks the same rotation either way.  What the yaw does change is the grasp
    itself -- the fingers have to close square onto two opposite lateral faces,
    and being out by the yaw angle means squeezing the die on its corners.  That
    is the failure the grasp frame exists to prevent, so it is what is asserted
    here.
    """
    for yaw_deg in (10.0, 30.0, 44.0):
        yaw = np.deg2rad(yaw_deg)
        yaw_matrix = yaw_rotation(yaw)

        # Face selection: identical plans with and without the conversion.
        for orientation in all_orientations():
            base_orientation = yaw_matrix @ orientation
            normals_base = {f: base_orientation @ n for f, n in FACE_NORMALS.items()}
            for target in range(1, 7):
                with_frame = plan_from_normals(
                    normals_in_grasp_frame(normals_base, yaw), target
                )
                without_frame = plan_from_normals(normals_base, target)
                assert len(with_frame) == len(without_frame)

        # Grasp geometry: the closing direction is only square on the die's
        # lateral faces when the yaw is taken into account.
        for axis in ("x", "y"):
            aligned = matrix_from_quaternion(grasp_orientation(axis, yaw))[:, 1]
            ignored = matrix_from_quaternion(grasp_orientation(axis, 0.0))[:, 1]
            lateral_normal = yaw_matrix @ AXIS_VECTORS[axis]

            assert abs(float(np.dot(aligned, lateral_normal))) == pytest.approx(1.0, abs=1e-9)
            misalignment = np.degrees(
                np.arccos(np.clip(abs(float(np.dot(ignored, lateral_normal))), 0.0, 1.0))
            )
            assert misalignment == pytest.approx(yaw_deg, abs=1e-6)


def test_nearest_cube_rotation_snaps_noisy_measurements():
    """A noisy or askew measurement must still yield a usable plan, not a crash."""
    from dice_task.die_model import nearest_cube_rotation

    rng = np.random.default_rng(11)
    for orientation in all_orientations():
        assert np.allclose(nearest_cube_rotation(orientation), orientation)

        wobbled = yaw_rotation(np.deg2rad(rng.uniform(-20, 20))) @ orientation
        snapped = nearest_cube_rotation(wobbled)
        assert np.allclose(snapped @ snapped.T, np.eye(3), atol=1e-9)
        assert up_face(snapped) == up_face(orientation)


# --------------------------------------------------------------------------- #
# Leaning the grasp so a 90-degree flip never ends horizontal
# --------------------------------------------------------------------------- #

TILT = np.deg2rad(45.0)


def _approach(quaternion):
    return matrix_from_quaternion(quaternion)[:, 2]


def _closing(quaternion, close_axis="y"):
    column = 1 if close_axis == "y" else 0
    return matrix_from_quaternion(quaternion)[:, column]


@pytest.mark.parametrize("grasp_axis", ["x", "y"])
@pytest.mark.parametrize("turns", [1, -1])
@pytest.mark.parametrize("yaw_deg", [0.0, 17.0, -33.0])
def test_a_leaning_grasp_is_never_horizontal_at_either_end(grasp_axis, turns, yaw_deg):
    """The bug this fixes: a straight-down grasp finishes a flip pointing sideways.

    A horizontal gripper cannot be lowered to the board -- its own body arrives
    first -- which is why MoveIt's Cartesian planner stopped three quarters of
    the way through the descent and the run aborted with error 99999.
    """
    yaw = np.deg2rad(yaw_deg)
    tilt = tilt_for_turn(turns, TILT)
    start = grasp_orientation(grasp_axis, yaw, "y", tilt)
    pivot = np.array([0.1, 0.5, 0.15])
    _, finish = rotate_about_axis(pivot, start, grasp_axis, turns, pivot, yaw)

    # "Pointing down" is a negative Z component of the approach axis.
    assert _approach(start)[2] == pytest.approx(-np.cos(TILT), abs=1e-9)
    assert _approach(finish)[2] == pytest.approx(-np.cos(TILT), abs=1e-9)

    # ...whereas the straight-down grasp it replaces does end horizontal.
    flat = grasp_orientation(grasp_axis, yaw, "y", 0.0)
    _, flat_finish = rotate_about_axis(pivot, flat, grasp_axis, turns, pivot, yaw)
    assert _approach(flat_finish)[2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("grasp_axis", ["x", "y"])
@pytest.mark.parametrize("turns", [1, -1])
@pytest.mark.parametrize("close_axis", ["x", "y"])
def test_leaning_does_not_move_the_fingers_off_their_faces(grasp_axis, turns, close_axis):
    """Leaning may only roll the gripper about the axis the fingers close along."""
    yaw = 0.37
    tilt = tilt_for_turn(turns, TILT)
    upright = grasp_orientation(grasp_axis, yaw, close_axis, 0.0)
    leaning = grasp_orientation(grasp_axis, yaw, close_axis, tilt)

    wanted = yaw_rotation(yaw) @ AXIS_VECTORS[grasp_axis]
    for quaternion in (upright, leaning):
        closing = _closing(quaternion, close_axis)
        assert abs(float(np.dot(closing, wanted))) == pytest.approx(1.0, abs=1e-9)
        assert closing[2] == pytest.approx(0.0, abs=1e-9)

    # The turn itself must leave the fingers where they are, lean or no lean.
    pivot = np.array([0.0, 0.5, 0.15])
    _, finish = rotate_about_axis(pivot, leaning, grasp_axis, turns, pivot, yaw)
    assert abs(float(np.dot(_closing(finish, close_axis), wanted))) == pytest.approx(1.0, abs=1e-9)


def test_tilt_for_turn_centres_the_excursion_on_vertical():
    for magnitude in (0.0, 0.3, TILT):
        assert tilt_for_turn(1, magnitude) == pytest.approx(-magnitude)
        assert tilt_for_turn(-1, magnitude) == pytest.approx(magnitude)


# --------------------------------------------------------------------------- #
# Choosing between the two rolls of the same grasp
# --------------------------------------------------------------------------- #


def test_the_flipped_grasp_grips_the_same_faces_from_the_same_side():
    for grasp_axis in ("x", "y"):
        for tilt in (0.0, -TILT, TILT):
            quaternion = grasp_orientation(grasp_axis, 0.4, "y", tilt)
            flipped = flip_about_approach(quaternion)
            assert not np.allclose(quaternion, flipped)
            # Same approach direction...
            assert np.allclose(_approach(flipped), _approach(quaternion), atol=1e-9)
            # ...and the same finger axis, just with the fingers swapped over.
            assert float(np.dot(_closing(flipped), _closing(quaternion))) == pytest.approx(
                -1.0, abs=1e-9
            )
            assert quaternion_distance(flipped, quaternion) == pytest.approx(np.pi, abs=1e-9)


def test_nearest_equivalent_grasp_saves_the_wrist_a_half_turn():
    quaternion = grasp_orientation("x", 0.2, "y", -TILT)
    flipped = flip_about_approach(quaternion)

    # Asked to come from where the flipped grasp already is, it must choose it.
    chosen = nearest_equivalent_grasp(quaternion, flipped)
    assert quaternion_distance(chosen, flipped) == pytest.approx(0.0, abs=1e-9)

    # Asked from the original, it keeps the original.
    assert quaternion_distance(
        nearest_equivalent_grasp(quaternion, quaternion), quaternion
    ) == pytest.approx(0.0, abs=1e-9)

    # With nothing to compare against, behaviour is unchanged.
    assert np.allclose(nearest_equivalent_grasp(quaternion, None), quaternion)


def test_nearest_equivalent_grasp_never_makes_the_wrist_travel_further():
    rng = np.random.default_rng(5)
    for _ in range(200):
        quaternion = grasp_orientation(
            "x" if rng.random() < 0.5 else "y", float(rng.uniform(-0.8, 0.8)), "y", -TILT
        )
        reference = quaternion_from_matrix(
            np.linalg.qr(rng.normal(size=(3, 3)))[0] * np.sign(
                np.linalg.det(np.linalg.qr(rng.normal(size=(3, 3)))[0])
            )
        )
        chosen = nearest_equivalent_grasp(quaternion, reference)
        assert quaternion_distance(chosen, reference) <= quaternion_distance(
            quaternion, reference
        ) + 1e-9


# --------------------------------------------------------------------------- #
# Backing off along the gripper's own axis
# --------------------------------------------------------------------------- #


def test_approach_offset_follows_the_tool_axis_not_the_vertical():
    upright = grasp_orientation("x", 0.0, "y", 0.0)
    assert np.allclose(approach_offset(upright, 0.10), [0.0, 0.0, 0.10], atol=1e-9)

    leaning = grasp_orientation("x", 0.0, "y", -TILT)
    offset = approach_offset(leaning, 0.10)
    assert float(np.linalg.norm(offset)) == pytest.approx(0.10, abs=1e-9)
    # Still retreats upwards, but no longer straight up: it slides the fingers
    # back along their own length instead of dragging one over the die.
    assert offset[2] == pytest.approx(0.10 * np.cos(TILT), abs=1e-9)
    assert abs(offset[1]) == pytest.approx(0.10 * np.sin(TILT), abs=1e-9)
    assert np.allclose(offset, -0.10 * _approach(leaning), atol=1e-9)
