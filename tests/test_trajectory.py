"""Tests for the trajectory generation described in the DRIMS planning lecture."""

import numpy as np
import pytest

from dice_task.grasping import matrix_from_quaternion, quaternion_from_matrix
from dice_task.trajectory import (
    BlendedTrajectory,
    CartesianPath,
    CartesianSegment,
    CartesianTrajectory,
    LinearPath,
    TrapezoidalProfile,
    blend_waypoints,
    quaternion_angle,
    slerp,
)


# --------------------------------------------------------------------------- #
# Trapezoidal velocity profile
# --------------------------------------------------------------------------- #


def test_worked_example_from_the_lecture():
    """L = 2 m rest-to-rest, unit v_max and a_max -> 3 s, exactly as plotted."""
    profile = TrapezoidalProfile(length=2.0, v_max=1.0, a_max=1.0)
    assert profile.duration == pytest.approx(3.0)
    assert profile.t_acc == pytest.approx(1.0)
    assert profile.t_flat == pytest.approx(1.0)
    assert profile.v_peak == pytest.approx(1.0)

    assert profile.position(0.0) == pytest.approx(0.0)
    assert profile.position(1.0) == pytest.approx(0.5)
    assert profile.position(2.0) == pytest.approx(1.5)
    assert profile.position(3.0) == pytest.approx(2.0)
    assert profile.velocity(1.5) == pytest.approx(1.0)
    assert profile.velocity(3.0) == pytest.approx(0.0)


def test_profile_reaches_exactly_the_requested_length():
    for length in (0.001, 0.05, 0.4, 2.0, 17.0):
        for v_max, a_max in ((0.25, 0.5), (1.0, 1.0), (2.0, 0.3)):
            profile = TrapezoidalProfile(length, v_max, a_max)
            assert profile.position(profile.duration) == pytest.approx(length, abs=1e-9)
            assert profile.position(0.0) == pytest.approx(0.0)


def test_profile_starts_and_ends_at_rest():
    for length in (0.01, 0.5, 5.0):
        profile = TrapezoidalProfile(length, 0.5, 1.0)
        assert profile.velocity(0.0) == pytest.approx(0.0)
        assert profile.velocity(profile.duration) == pytest.approx(0.0, abs=1e-9)


def test_profile_never_exceeds_its_limits():
    rng = np.random.default_rng(3)
    for _ in range(200):
        length = float(rng.uniform(0.001, 5.0))
        v_max = float(rng.uniform(0.05, 2.0))
        a_max = float(rng.uniform(0.05, 4.0))
        profile = TrapezoidalProfile(length, v_max, a_max)
        for t in np.linspace(0.0, profile.duration, 200):
            assert abs(profile.velocity(t)) <= v_max + 1e-9
            assert abs(profile.acceleration(t)) <= a_max + 1e-9


def test_short_paths_use_the_triangular_profile():
    """Below v_max^2 / a_max the cruise phase disappears."""
    v_max, a_max = 1.0, 1.0
    short = TrapezoidalProfile(0.5, v_max, a_max)  # ramp distance is 1.0
    assert short.t_flat == pytest.approx(0.0)
    assert short.v_peak < v_max
    assert short.duration == pytest.approx(2.0 * np.sqrt(0.5))
    assert short.position(short.duration) == pytest.approx(0.5)

    long = TrapezoidalProfile(2.0, v_max, a_max)
    assert long.t_flat > 0.0
    assert long.v_peak == pytest.approx(v_max)


def test_velocity_is_the_derivative_of_position():
    profile = TrapezoidalProfile(1.3, 0.4, 0.9)
    dt = 1e-6
    for t in np.linspace(dt, profile.duration - dt, 50):
        numeric = (profile.position(t + dt) - profile.position(t - dt)) / (2 * dt)
        assert numeric == pytest.approx(profile.velocity(t), abs=1e-4)


def test_position_is_monotonic():
    profile = TrapezoidalProfile(2.0, 0.7, 1.1)
    samples = [profile.position(t) for t in np.linspace(0, profile.duration, 500)]
    assert all(b >= a - 1e-12 for a, b in zip(samples, samples[1:]))


def test_zero_length_profile_is_instantaneous():
    profile = TrapezoidalProfile(0.0, 1.0, 1.0)
    assert profile.duration == 0.0
    assert profile.position(0.0) == 0.0
    assert profile.normalised(0.0) == 1.0


def test_profile_rejects_nonsense():
    with pytest.raises(ValueError):
        TrapezoidalProfile(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        TrapezoidalProfile(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        TrapezoidalProfile(1.0, 1.0, -2.0)


def test_stretching_preserves_length_and_slows_down():
    profile = TrapezoidalProfile(1.0, 1.0, 1.0)
    slower = profile.stretched_to(2.0 * profile.duration)
    assert slower.duration == pytest.approx(2.0 * profile.duration)
    assert slower.position(slower.duration) == pytest.approx(1.0)
    assert slower.v_peak < profile.v_peak


def test_stretching_cannot_speed_a_profile_up_past_its_limits():
    profile = TrapezoidalProfile(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        profile.stretched_to(0.5 * profile.duration)


# --------------------------------------------------------------------------- #
# LIN path
# --------------------------------------------------------------------------- #


def test_linear_path_endpoints_and_straightness():
    path = LinearPath([0.4, 0.0, 0.3], [0.6, 0.2, 0.1])
    assert np.allclose(path.at(0.0), [0.4, 0.0, 0.3])
    assert np.allclose(path.at(1.0), [0.6, 0.2, 0.1])
    assert path.length == pytest.approx(np.linalg.norm([0.2, 0.2, -0.2]))

    # Every sample lies on the segment.
    for u in np.linspace(0, 1, 25):
        point = path.at(u)
        along = np.dot(point - path.start, path.direction)
        assert np.allclose(point, path.start + along * path.direction, atol=1e-12)


def test_degenerate_linear_path():
    path = LinearPath([0.5, 0.1, 0.2], [0.5, 0.1, 0.2])
    assert path.length == 0.0
    assert np.allclose(path.at(0.7), [0.5, 0.1, 0.2])


# --------------------------------------------------------------------------- #
# SLERP
# --------------------------------------------------------------------------- #


def _quaternion_about(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])


def test_slerp_hits_its_endpoints():
    a = _quaternion_about([0, 0, 1], 0.0)
    b = _quaternion_about([0, 1, 0], 1.2)
    assert np.allclose(slerp(a, b, 0.0), a, atol=1e-12)
    assert np.allclose(np.abs(slerp(a, b, 1.0)), np.abs(b), atol=1e-12)


def test_slerp_stays_on_the_unit_sphere():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        for u in np.linspace(0, 1, 20):
            assert np.linalg.norm(slerp(a, b, u)) == pytest.approx(1.0)


def test_slerp_has_constant_angular_rate():
    """Equal parameter steps must produce equal rotation increments."""
    a = _quaternion_about([0, 0, 1], 0.0)
    b = _quaternion_about([0.3, 0.5, 0.8], 2.0)
    steps = np.linspace(0, 1, 21)
    angles = [
        quaternion_angle(slerp(a, b, u), slerp(a, b, v))
        for u, v in zip(steps, steps[1:])
    ]
    assert np.std(angles) < 1e-9


def test_slerp_takes_the_short_way_round():
    """A quaternion and its negation are the same rotation."""
    a = _quaternion_about([0, 0, 1], 0.0)
    b = -_quaternion_about([0, 0, 1], 0.4)  # same rotation, negated
    midpoint = slerp(a, b, 0.5)
    assert quaternion_angle(a, midpoint) == pytest.approx(0.2, abs=1e-9)
    assert quaternion_angle(a, b) == pytest.approx(0.4, abs=1e-9)


def test_slerp_matches_matrix_interpolation():
    """Cross-check against the axis-angle formulation from the same slide."""
    start = _quaternion_about([0, 0, 1], 0.2)
    end = _quaternion_about([1, 0, 0], 1.1)
    r_a = matrix_from_quaternion(start)
    r_b = matrix_from_quaternion(end)
    relative = r_a.T @ r_b
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / (2 * np.sin(angle))

    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        partial = _quaternion_about(axis, u * angle)
        expected = quaternion_from_matrix(r_a @ matrix_from_quaternion(partial))
        actual = slerp(start, end, u)
        assert quaternion_angle(expected, actual) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Synchronised Cartesian segments
# --------------------------------------------------------------------------- #


def test_segment_reaches_both_targets_together():
    """Translation and rotation must finish at the same instant."""
    segment = CartesianSegment(
        start_position=[0.5, 0.0, 0.3],
        end_position=[0.6, 0.1, 0.2],
        start_quaternion=_quaternion_about([0, 0, 1], 0.0),
        end_quaternion=_quaternion_about([1, 0, 0], np.pi / 2),
    )
    position, orientation = segment.pose_at(segment.duration)
    assert np.allclose(position, [0.6, 0.1, 0.2], atol=1e-9)
    assert quaternion_angle(orientation, segment.end_quaternion) == pytest.approx(
        0.0, abs=1e-9
    )


def test_segment_duration_is_set_by_the_binding_constraint():
    common = dict(
        start_position=[0.0, 0.0, 0.0],
        start_quaternion=[0, 0, 0, 1.0],
        v_max=0.25,
        a_max=0.5,
        w_max=1.5,
        alpha_max=3.0,
    )
    # A long move with almost no rotation: translation dominates.
    translation_bound = CartesianSegment(
        end_position=[1.0, 0, 0], end_quaternion=_quaternion_about([0, 0, 1], 0.01), **common
    )
    assert translation_bound.duration == pytest.approx(
        translation_bound.translation.duration
    )

    # A tiny move with a big turn: rotation dominates.
    rotation_bound = CartesianSegment(
        end_position=[0.001, 0, 0], end_quaternion=_quaternion_about([0, 0, 1], 3.0), **common
    )
    assert rotation_bound.duration == pytest.approx(rotation_bound.rotation.duration)


def test_segment_respects_limits_throughout():
    segment = CartesianSegment(
        start_position=[0.4, 0.0, 0.3],
        end_position=[0.7, 0.2, 0.1],
        start_quaternion=_quaternion_about([0, 0, 1], 0.0),
        end_quaternion=_quaternion_about([0, 1, 0], 2.4),
        v_max=0.25,
        a_max=0.5,
        w_max=1.5,
        alpha_max=3.0,
    )
    for t in np.linspace(0, segment.duration, 300):
        linear, angular = segment.twist_at(t)
        assert linear <= 0.25 + 1e-9
        assert angular <= 1.5 + 1e-9


def test_segment_path_is_straight():
    segment = CartesianSegment(
        start_position=[0.4, 0.0, 0.3],
        end_position=[0.7, 0.2, 0.1],
        start_quaternion=[0, 0, 0, 1.0],
        end_quaternion=_quaternion_about([1, 0, 0], 0.6),
    )
    start, end = segment.path.start, segment.path.end
    direction = (end - start) / np.linalg.norm(end - start)
    for t in np.linspace(0, segment.duration, 60):
        position, _ = segment.pose_at(t)
        offset = position - start
        perpendicular = offset - np.dot(offset, direction) * direction
        assert np.linalg.norm(perpendicular) < 1e-12


# --------------------------------------------------------------------------- #
# Multi-segment trajectories
# --------------------------------------------------------------------------- #


def _pick_place_waypoints():
    down = _quaternion_about([1, 0, 0], np.pi)
    return [
        ([0.5, 0.0, 0.30], down),
        ([0.5, 0.0, 0.05], down),
        ([0.5, 0.0, 0.25], down),
        ([0.3, 0.2, 0.25], down),
        ([0.3, 0.2, 0.05], down),
    ]


def test_trajectory_is_continuous_across_segments():
    trajectory = CartesianTrajectory.through(_pick_place_waypoints())
    dt = 1e-4
    for boundary in trajectory.starts[1:]:
        before, _ = trajectory.pose_at(boundary - dt)
        after, _ = trajectory.pose_at(boundary + dt)
        assert np.linalg.norm(after - before) < 1e-3


def test_trajectory_passes_through_every_waypoint():
    waypoints = _pick_place_waypoints()
    trajectory = CartesianTrajectory.through(waypoints)
    for start, (expected, _) in zip(trajectory.starts, waypoints):
        actual, _ = trajectory.pose_at(start)
        assert np.allclose(actual, expected, atol=1e-9)
    final, _ = trajectory.pose_at(trajectory.duration)
    assert np.allclose(final, waypoints[-1][0], atol=1e-9)


def test_sampling_covers_the_whole_trajectory():
    trajectory = CartesianTrajectory.through(_pick_place_waypoints())
    samples = trajectory.sample(0.05)
    assert samples[0][0] == pytest.approx(0.0)
    assert samples[-1][0] == pytest.approx(trajectory.duration)
    times = [t for t, _, _ in samples]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_trajectory_rejects_bad_input():
    with pytest.raises(ValueError):
        CartesianTrajectory.through([([0, 0, 0], [0, 0, 0, 1])])
    with pytest.raises(ValueError):
        CartesianTrajectory.through(_pick_place_waypoints()).sample(0.0)


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #


def test_blending_keeps_the_endpoints():
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    blended = blend_waypoints(points, 0.2)
    assert np.allclose(blended[0], points[0])
    assert np.allclose(blended[-1], points[-1])


def test_blending_cuts_the_corner():
    """The blended path must be shorter than the sharp one."""
    points = [np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([1.0, 1.0, 0])]
    sharp = sum(
        float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])
    )
    blended = blend_waypoints(points, 0.3)
    rounded = sum(
        float(np.linalg.norm(b - a)) for a, b in zip(blended, blended[1:])
    )
    assert rounded < sharp
    assert rounded > 0.9 * sharp  # but it is a corner cut, not a shortcut


def test_blending_removes_the_full_stop_at_corners():
    """The point of blending: no stop at interior waypoints, so a faster cycle.

    The comparison has to be like for like. Subdividing a path and still
    stopping at every point makes it *slower*, which is what issuing one
    move_to_pose per waypoint does. The gain comes from blending the corners
    *and* running a single profile across the whole path.
    """
    waypoints = _pick_place_waypoints()
    orientation = waypoints[0][1]
    positions = [p for p, _ in waypoints]

    stop_at_each = CartesianTrajectory.through([(p, orientation) for p in positions])
    blended = BlendedTrajectory(
        [(p, orientation) for p in blend_waypoints(positions, 0.04)],
        lateral_a_max=2.0,
    )
    assert blended.duration < stop_at_each.duration


def test_without_blending_there_is_nothing_to_gain():
    """An un-blended pick-and-place is *all* hard corners, so it cannot be sped up.

    Every interior waypoint of the raw path turns by 90 or 180 degrees, so the
    trajectory has to stop at each one and matches the stop-at-every-waypoint
    baseline exactly. The saving comes from blending, not from the continuous
    traversal on its own -- worth pinning down, because it is tempting to credit
    the wrong half of the change.
    """
    waypoints = _pick_place_waypoints()
    stop_at_each = CartesianTrajectory.through(waypoints)
    continuous = BlendedTrajectory(waypoints)
    assert continuous.stop_count == len(stop_at_each.segments) + 1
    assert continuous.duration == pytest.approx(stop_at_each.duration, rel=1e-9)


def test_blended_path_endpoints_and_waypoint_order():
    waypoints = _pick_place_waypoints()
    trajectory = BlendedTrajectory(waypoints)
    start, _ = trajectory.pose_at(0.0)
    end, _ = trajectory.pose_at(trajectory.duration)
    assert np.allclose(start, waypoints[0][0], atol=1e-9)
    assert np.allclose(end, waypoints[-1][0], atol=1e-9)


def test_blended_trajectory_does_not_stop_on_a_smooth_path():
    orientation = _quaternion_about([1, 0, 0], np.pi)
    smooth = blend_waypoints(
        [[0.5, 0.0, 0.25], [0.4, 0.15, 0.25], [0.3, 0.2, 0.25]], 0.03
    )
    trajectory = BlendedTrajectory([(p, orientation) for p in smooth])
    assert trajectory.stop_count == 2, "a smooth path should stop only at its ends"
    assert trajectory.speed_at(0.0) == pytest.approx(0.0)
    assert trajectory.speed_at(trajectory.duration) == pytest.approx(0.0, abs=1e-9)
    interior = [
        trajectory.speed_at(t)
        for t in np.linspace(0.05 * trajectory.duration, 0.95 * trajectory.duration, 50)
    ]
    assert min(interior) > 1e-3, "the tool came to a stop mid-path"


def test_a_reversal_forces_a_stop_no_matter_the_blend_radius():
    """Descend to grasp then climb back up: a 180-degree turn cannot be rounded."""
    orientation = _quaternion_about([1, 0, 0], np.pi)
    reversal = [[0.5, 0, 0.25], [0.5, 0, 0.05], [0.5, 0, 0.25]]
    for radius in (0.0, 0.01, 0.05, 0.5):
        points = blend_waypoints(reversal, radius)
        trajectory = BlendedTrajectory([(p, orientation) for p in points])
        assert trajectory.stop_count >= 3, f"radius {radius} pretended to blend a reversal"


def test_blending_reduces_the_number_of_stops():
    waypoints = _pick_place_waypoints()
    orientation = waypoints[0][1]
    positions = [p for p, _ in waypoints]

    stop_at_each = CartesianTrajectory.through([(p, orientation) for p in positions])
    blended = BlendedTrajectory(
        [(p, orientation) for p in blend_waypoints(positions, 0.04)],
        lateral_a_max=2.0,
    )
    # Five waypoints means four full stops the old way; blending leaves only the
    # unavoidable reversal at the grasp.
    assert len(stop_at_each.segments) == 4
    assert blended.stop_count < len(stop_at_each.segments) + 1


def test_sharp_corners_admit_no_speed_but_blended_ones_do():
    """The physical reason blending is needed at all."""
    orientation = _quaternion_about([1, 0, 0], np.pi)
    corner = [[0.0, 0, 0], [0.3, 0, 0], [0.3, 0.3, 0]]

    sharp_path = CartesianPath([(p, orientation) for p in corner])
    assert sharp_path.hard_corners() == [1]
    assert sharp_path.min_turn_radius() == pytest.approx(0.0, abs=1e-9)

    rounded = blend_waypoints(corner, 0.05)
    rounded_path = CartesianPath([(p, orientation) for p in rounded])
    assert rounded_path.hard_corners() == [], "blending left a hard corner"
    assert rounded_path.min_turn_radius() > 1e-3

    sharp = BlendedTrajectory([(p, orientation) for p in corner], lateral_a_max=2.0)
    blended = BlendedTrajectory([(p, orientation) for p in rounded], lateral_a_max=2.0)

    # The sharp corner is handled by stopping at it: two straight pieces, each
    # of them curvature-free, so there is no corner speed to limit.
    assert sharp.stop_count == 3
    assert sharp.corner_speed_limit is None
    # The blended one is a single smooth run whose speed is capped by the bend.
    assert blended.stop_count == 2
    assert blended.corner_speed_limit is not None and blended.corner_speed_limit > 0.05
    assert blended.duration < sharp.duration


def test_lateral_acceleration_cap_is_applied():
    orientation = _quaternion_about([1, 0, 0], np.pi)
    corner = blend_waypoints([[0.0, 0, 0], [0.4, 0, 0], [0.4, 0.4, 0]], 0.03)
    fast = BlendedTrajectory([(p, orientation) for p in corner], v_max=1.0)
    limited = BlendedTrajectory(
        [(p, orientation) for p in corner], v_max=1.0, lateral_a_max=1.0
    )
    assert limited.v_max < fast.v_max
    assert limited.duration > fast.duration


def test_blend_radius_is_clipped_to_the_shorter_segment():
    """An over-large radius must not let the path skip a waypoint."""
    points = [[0, 0, 0], [0.05, 0, 0], [0.05, 1.0, 0]]
    blended = blend_waypoints(points, 10.0)
    corner = np.array([0.05, 0, 0])
    assert min(float(np.linalg.norm(np.asarray(p) - corner)) for p in blended) < 0.03


def test_blending_is_a_no_op_without_corners():
    points = [[0, 0, 0], [1, 0, 0]]
    assert len(blend_waypoints(points, 0.5)) == 2
    assert len(blend_waypoints(points, 0.0)) == 2
