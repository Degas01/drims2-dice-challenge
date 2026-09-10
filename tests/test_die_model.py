"""Exhaustive tests for the die model and the re-orientation planners.

These run without ROS: ``pytest tests/test_die_model.py``.
"""

import itertools

import numpy as np
import pytest

from dice_task.die_model import (
    FACE_NORMALS,
    PRIMITIVES,
    DeductivePolicy,
    Regrasp,
    apply,
    face_towards,
    opposite,
    orientation_from_two_faces,
    orientation_from_up_and_yaw,
    plan_exact,
    plan_from_normals,
    simulate_blind,
    up_face,
)


def all_orientations():
    """The 24 rotations of the cube, generated from the primitive moves."""
    seen = {}
    frontier = [np.eye(3)]
    seen[tuple(np.eye(3).flatten().astype(int))] = np.eye(3)
    while frontier:
        cur = frontier.pop()
        for move in PRIMITIVES:
            nxt = apply(cur, move)
            key = tuple(np.round(nxt.flatten()).astype(int))
            if key not in seen:
                seen[key] = nxt
                frontier.append(nxt)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Geometry invariants
# --------------------------------------------------------------------------- #


def test_opposite_faces_sum_to_seven():
    for face in range(1, 7):
        assert face + opposite(face) == 7
        assert np.allclose(FACE_NORMALS[face], -FACE_NORMALS[opposite(face)])


def test_face_normals_are_orthonormal_axes():
    for face, normal in FACE_NORMALS.items():
        assert pytest.approx(1.0) == float(np.linalg.norm(normal))


def test_cube_rotation_group_has_24_elements():
    assert len(all_orientations()) == 24


def test_every_orientation_is_a_proper_rotation():
    for rot in all_orientations():
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert pytest.approx(1.0, abs=1e-9) == float(np.linalg.det(rot))


def test_each_face_is_up_in_exactly_four_orientations():
    counts = {face: 0 for face in range(1, 7)}
    for rot in all_orientations():
        counts[up_face(rot)] += 1
    assert counts == {face: 4 for face in range(1, 7)}


# --------------------------------------------------------------------------- #
# Exact planner
# --------------------------------------------------------------------------- #


def test_exact_plan_reaches_target_from_every_orientation():
    for rot in all_orientations():
        for target in range(1, 7):
            plan = plan_exact(rot, target)
            state = rot
            for move in plan:
                state = apply(state, move)
            assert up_face(state) == target, (
                f"plan {plan} failed from up={up_face(rot)} to {target}"
            )


def test_exact_plan_is_minimal():
    """0 moves when already up, 1 for a lateral face, 2 for the bottom face."""
    for rot in all_orientations():
        current = up_face(rot)
        for target in range(1, 7):
            plan = plan_exact(rot, target)
            if target == current:
                assert len(plan) == 0
            elif target == opposite(current):
                assert len(plan) == 2
            else:
                assert len(plan) == 1


def test_exact_plan_never_exceeds_two_moves():
    worst = max(
        len(plan_exact(rot, target))
        for rot in all_orientations()
        for target in range(1, 7)
    )
    assert worst == 2


def test_plan_rejects_invalid_target():
    with pytest.raises(ValueError):
        plan_exact(np.eye(3), 7)
    with pytest.raises(ValueError):
        plan_exact(np.eye(3), 0)


# --------------------------------------------------------------------------- #
# Planning from measured normals (what the TF lookup gives us)
# --------------------------------------------------------------------------- #


def _normals_in_world(rot):
    return {face: rot @ n for face, n in FACE_NORMALS.items()}


def test_plan_from_normals_matches_plan_exact():
    for rot in all_orientations():
        normals = _normals_in_world(rot)
        for target in range(1, 7):
            assert len(plan_from_normals(normals, target)) == len(
                plan_exact(rot, target)
            )


def test_plan_from_normals_tolerates_tf_noise():
    rng = np.random.default_rng(20260909)
    for rot in all_orientations():
        for target in range(1, 7):
            noisy = {
                face: n + rng.normal(scale=0.02, size=3)
                for face, n in _normals_in_world(rot).items()
            }
            plan = plan_from_normals(noisy, target)
            state = rot
            for move in plan:
                state = apply(state, move)
            assert up_face(state) == target


def test_plan_from_normals_works_with_only_three_faces_visible():
    """Opposite faces are inferred, so three normals are enough."""
    for rot in all_orientations():
        normals = _normals_in_world(rot)
        partial = {face: normals[face] for face in (2, 4, 1)}  # opposites of 5, 3, 6
        for target in range(1, 7):
            plan = plan_from_normals(partial, target)
            state = rot
            for move in plan:
                state = apply(state, move)
            assert up_face(state) == target


# --------------------------------------------------------------------------- #
# Deductive (vision-only) policy
# --------------------------------------------------------------------------- #


def test_deductive_policy_always_converges():
    for rot in all_orientations():
        for target in range(1, 7):
            for axis in ("x", "y"):
                moves = simulate_blind(rot, target, first_axis=axis)
                state = rot
                for move in moves:
                    state = apply(state, move)
                assert up_face(state) == target


def test_deductive_policy_worst_case_is_bounded():
    worst = max(
        len(simulate_blind(rot, target, first_axis=axis))
        for rot in all_orientations()
        for target in range(1, 7)
        for axis in ("x", "y")
    )
    assert worst <= 3, f"deductive policy worst case regressed to {worst} moves"


def test_deductive_policy_cost_distribution_is_exactly_as_documented():
    """Pins the numbers quoted in the README so they cannot silently drift."""
    import collections

    for axis in ("x", "y"):
        lengths = [
            len(simulate_blind(rot, target, first_axis=axis))
            for rot in all_orientations()
            for target in range(1, 7)
        ]
        # 0 already up; 1 the probed side; 2 the bottom or a grasp-axis face;
        # 3 the face opposite the probed side.
        assert collections.Counter(lengths) == {0: 24, 1: 24, 2: 72, 3: 24}
        assert sum(lengths) / len(lengths) == pytest.approx(5 / 3)


def test_exact_planner_beats_deductive_planner():
    """The whole point of using the face TFs: fewer re-grasps, hard cap of two."""
    exact = [
        len(plan_exact(rot, target))
        for rot in all_orientations()
        for target in range(1, 7)
    ]
    deduced = [
        len(simulate_blind(rot, target))
        for rot in all_orientations()
        for target in range(1, 7)
    ]
    assert sum(exact) / len(exact) == pytest.approx(1.0)
    assert max(exact) == 2 < max(deduced)


def test_deductive_policy_stops_immediately_when_already_correct():
    for rot in all_orientations():
        assert simulate_blind(rot, up_face(rot)) == []


def test_target_on_the_bottom_takes_two_turns_about_one_axis():
    """Stage two of the user-facing rule: 180 degrees, same axis, both quarters."""
    for rot in all_orientations():
        target = opposite(up_face(rot))
        moves = simulate_blind(rot, target)
        assert len(moves) == 2
        assert moves[0].axis == moves[1].axis
        assert moves[0].quarter_turns == moves[1].quarter_turns


def test_one_probe_is_enough_to_know_the_whole_die():
    """After the single 90-degree probe the policy holds the true orientation."""
    for rot in all_orientations():
        for target in range(1, 7):
            if target in (up_face(rot), opposite(up_face(rot))):
                continue
            policy = DeductivePolicy(target)
            probe = policy.observe(up_face(rot))
            assert policy.orientation is None, "nothing is known before the probe"
            after = apply(rot, probe)
            policy.observe(up_face(after))
            if target == up_face(after):
                continue  # the probe happened to finish the job
            assert policy.orientation is not None
            assert np.allclose(policy.orientation, apply(after, plan_exact(after, target)[0]))


def test_deduction_survives_the_die_being_nudged():
    """A reading that no orientation explains must not poison the model."""
    policy = DeductivePolicy(target_face=2)
    policy.observe(5)
    # 5 and its opposite 2 cannot both be lateral faces after one quarter turn.
    policy.observe(5)
    assert policy.orientation is None
    # ...and it keeps making legal progress rather than raising.
    assert policy.observe(3) is not None


def test_orientation_from_two_faces_is_unique_and_total():
    """Every (top, +Y) pair maps to exactly one orientation, and all 24 occur."""
    seen = set()
    for rot in all_orientations():
        top, side = up_face(rot), face_towards(rot, (0, 1, 0))
        assert (top, side) not in seen
        seen.add((top, side))
        recovered = orientation_from_two_faces(top, side, (0, 1, 0))
        assert recovered is not None
        assert np.allclose(recovered, rot)
    assert len(seen) == 24


def test_orientation_from_two_faces_rejects_impossible_pairs():
    for top in range(1, 7):
        assert orientation_from_two_faces(top, top, (0, 1, 0)) is None
        assert orientation_from_two_faces(top, opposite(top), (0, 1, 0)) is None


# --------------------------------------------------------------------------- #
# Primitive sanity
# --------------------------------------------------------------------------- #


def _faces_on_world_axis(rot, axis_vec):
    """The two faces whose normals point along +/- the given world axis."""
    return {
        face
        for face, normal in FACE_NORMALS.items()
        if abs(float(np.dot(rot @ normal, axis_vec))) > 0.9
    }


def test_quarter_turn_never_lifts_the_faces_on_the_grasp_axis():
    """Turning about world X cycles the other four faces and pins the +/-X pair.

    Which face IDs those are depends on the starting orientation -- they are the
    faces the *gripper fingers* are holding -- so the invariant is stated
    geometrically rather than with hard-coded face numbers.
    """
    for rot in all_orientations():
        for axis, axis_vec in (("x", np.array([1.0, 0, 0])), ("y", np.array([0, 1.0, 0]))):
            pinned = _faces_on_world_axis(rot, axis_vec)
            assert len(pinned) == 2
            assert sum(pinned) == 7, "the pinned pair must be opposite faces"

            state, reached = rot, set()
            for _ in range(4):
                state = apply(state, Regrasp(axis, 1))
                reached.add(up_face(state))
                assert _faces_on_world_axis(state, axis_vec) == pinned

            assert reached.isdisjoint(pinned)
            assert len(reached) == 4, "the other four faces each come up once"


def test_four_quarter_turns_return_to_start():
    for rot in all_orientations():
        for axis in ("x", "y"):
            state = rot
            for _ in range(4):
                state = apply(state, Regrasp(axis, 1))
            assert np.allclose(state, rot)


def test_opposite_signs_cancel():
    for rot in all_orientations():
        for axis in ("x", "y"):
            state = apply(apply(rot, Regrasp(axis, 1)), Regrasp(axis, -1))
            assert np.allclose(state, rot)
