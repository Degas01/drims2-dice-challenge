"""Tests for turning a Cartesian trajectory into a joint trajectory.

No ROS and no robot: the IK is a stub with a known closed form, so the tests can
check the pipeline's behaviour -- seeding, branch detection, failure reporting --
rather than a solver's numerics.
"""

import numpy as np
import pytest

from dice_task.cartesian_executor import (
    TrajectoryPlan,
    estimate_joint_velocities,
    plan_joint_trajectory,
)
from dice_task.trajectory import BlendedTrajectory, blend_waypoints


DOWN = np.array([1.0, 0.0, 0.0, 0.0])  # tool pointing at the table


def _planar_ik(position, quaternion, seed):
    """A two-link planar arm, with a genuine elbow-up / elbow-down choice."""
    del quaternion
    l1 = l2 = 0.6
    x, y = float(position[0]), float(position[1])
    reach = np.hypot(x, y)
    if reach > l1 + l2 or reach < 1e-6:
        return None

    cos_elbow = (reach**2 - l1**2 - l2**2) / (2 * l1 * l2)
    elbow = float(np.arccos(np.clip(cos_elbow, -1.0, 1.0)))
    branches = []
    for sign in (+1.0, -1.0):
        q2 = sign * elbow
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        branches.append([float(q1), float(q2)])

    if seed is None:
        return branches[0]
    # Pick whichever branch is closest to the seed: this is what a real solver
    # does with a seed, and what keeps the arm on one branch.
    return min(
        branches,
        key=lambda candidate: float(np.max(np.abs(np.asarray(candidate) - np.asarray(seed)))),
    )


def _always_fails(position, quaternion, seed):
    del position, quaternion, seed
    return None


def _fails_after(n):
    state = {"calls": 0}

    def solver(position, quaternion, seed):
        state["calls"] += 1
        if state["calls"] > n:
            return None
        return _planar_ik(position, quaternion, seed)

    return solver


def _straight_samples(count=25, start=(0.5, 0.4), end=(0.7, 0.2), duration=2.0):
    times = np.linspace(0.0, duration, count)
    xs = np.linspace(start[0], end[0], count)
    ys = np.linspace(start[1], end[1], count)
    return [
        (float(t), np.array([x, y, 0.2]), DOWN) for t, x, y in zip(times, xs, ys)
    ]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_plans_a_joint_trajectory_for_every_sample():
    samples = _straight_samples()
    plan = plan_joint_trajectory(samples, _planar_ik)
    assert plan.ok
    assert len(plan.waypoints) == len(samples)
    assert plan.duration == pytest.approx(samples[-1][0])


def test_times_are_preserved_and_monotonic():
    plan = plan_joint_trajectory(_straight_samples(), _planar_ik)
    times = [w.time for w in plan.waypoints]
    assert times[0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(times, times[1:]))


def test_joint_path_is_smooth_when_the_cartesian_path_is():
    plan = plan_joint_trajectory(_straight_samples(count=60), _planar_ik)
    assert plan.ok
    assert plan.max_joint_step < 0.1


def test_forward_kinematics_round_trip():
    """The IK solutions really do reach the requested Cartesian points."""
    samples = _straight_samples()
    plan = plan_joint_trajectory(samples, _planar_ik)
    l1 = l2 = 0.6
    for (_, position, _), waypoint in zip(samples, plan.waypoints):
        q1, q2 = waypoint.positions
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        assert np.allclose([x, y], position[:2], atol=1e-9)


# --------------------------------------------------------------------------- #
# Seeding and branch continuity
# --------------------------------------------------------------------------- #


def test_each_solution_seeds_the_next():
    """Without seeding the solver is free to flip branches between samples."""
    samples = _straight_samples(count=40)
    seeded = plan_joint_trajectory(samples, _planar_ik, seed=[0.4, 1.5])

    unseeded_solutions = [_planar_ik(p, q, None) for _, p, q in samples]
    unseeded_steps = max(
        float(np.max(np.abs(np.asarray(b) - np.asarray(a))))
        for a, b in zip(unseeded_solutions, unseeded_solutions[1:])
    )
    assert seeded.ok
    # Seeding must not make continuity worse, and it pins the chosen branch.
    assert seeded.max_joint_step <= unseeded_steps + 1e-9
    assert seeded.waypoints[0].positions[1] > 0, "seed should select elbow-up"


def test_starting_seed_selects_the_branch():
    samples = _straight_samples()
    up = plan_joint_trajectory(samples, _planar_ik, seed=[0.4, 1.5])
    down = plan_joint_trajectory(samples, _planar_ik, seed=[0.4, -1.5])
    assert up.waypoints[0].positions[1] > 0
    assert down.waypoints[0].positions[1] < 0


def test_branch_flip_is_detected_not_executed():
    """A discontinuous joint path must be reported, never handed to the arm."""

    def flips_halfway(position, quaternion, seed):
        del seed
        # Ignore the seed, so the solver is free to jump branches.
        solution = _planar_ik(position, quaternion, None)
        if solution is None:
            return None
        if position[0] > 0.6:
            return [solution[0], -solution[1]]
        return solution

    plan = plan_joint_trajectory(_straight_samples(), flips_halfway, max_joint_step=0.3)
    assert not plan.ok
    assert "branch change" in plan.failure
    assert "swing the arm" in plan.failure


def test_max_joint_step_is_configurable():
    def jumpy(position, quaternion, seed):
        del quaternion, seed
        return [float(position[0]), 0.0]

    samples = _straight_samples(count=3)
    assert plan_joint_trajectory(samples, jumpy, max_joint_step=1e-6).failure is not None
    assert plan_joint_trajectory(samples, jumpy, max_joint_step=10.0).ok


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_unreachable_sample_aborts_the_whole_trajectory():
    """Dropping the sample would silently straighten a deliberately bent path."""
    plan = plan_joint_trajectory(_straight_samples(), _fails_after(10))
    assert not plan.ok
    assert "no IK solution at sample 10" in plan.failure
    assert len(plan.waypoints) == 10  # what it got through, for diagnosis


def test_total_ik_failure_is_reported_clearly():
    plan = plan_joint_trajectory(_straight_samples(), _always_fails)
    assert not plan.ok
    assert "no IK solution at sample 0" in plan.failure


def test_out_of_reach_target_fails():
    far = [(0.0, np.array([5.0, 0.0, 0.2]), DOWN), (1.0, np.array([5.1, 0.0, 0.2]), DOWN)]
    assert not plan_joint_trajectory(far, _planar_ik).ok


def test_too_few_samples():
    plan = plan_joint_trajectory([(0.0, np.array([0.5, 0.4, 0.2]), DOWN)], _planar_ik)
    assert not plan.ok
    assert "at least two" in plan.failure


def test_empty_plan_is_not_ok():
    assert not TrajectoryPlan().ok
    assert TrajectoryPlan().duration == 0.0


# --------------------------------------------------------------------------- #
# Velocities
# --------------------------------------------------------------------------- #


def test_velocity_estimate_shape_and_magnitude():
    plan = plan_joint_trajectory(_straight_samples(count=30), _planar_ik)
    velocities = estimate_joint_velocities(plan)
    assert velocities.shape == (29, 2)
    assert np.all(np.isfinite(velocities))
    assert np.max(np.abs(velocities)) < 5.0


def test_velocity_estimate_handles_degenerate_plans():
    assert estimate_joint_velocities(TrajectoryPlan()).size == 0


# --------------------------------------------------------------------------- #
# Together with the trajectory generator
# --------------------------------------------------------------------------- #


def test_end_to_end_from_a_blended_trajectory():
    """Waypoints -> blend -> profile -> sample -> IK -> joint trajectory."""
    corner = [[0.5, 0.4, 0.2], [0.7, 0.4, 0.2], [0.7, 0.2, 0.2]]
    rounded = blend_waypoints(corner, 0.04)
    trajectory = BlendedTrajectory(
        [(p, DOWN) for p in rounded], v_max=0.25, a_max=0.5, lateral_a_max=2.0
    )
    samples = trajectory.sample(0.05)

    plan = plan_joint_trajectory(samples, _planar_ik, seed=[0.4, 1.5])
    assert plan.ok
    assert plan.duration == pytest.approx(trajectory.duration)
    assert plan.max_joint_step < 0.2

    velocities = estimate_joint_velocities(plan)
    assert np.max(np.abs(velocities)) < 3.0


def test_a_denser_sample_gives_a_smoother_joint_path():
    corner = blend_waypoints([[0.5, 0.4, 0.2], [0.7, 0.4, 0.2], [0.7, 0.2, 0.2]], 0.04)
    trajectory = BlendedTrajectory([(p, DOWN) for p in corner], lateral_a_max=2.0)
    coarse = plan_joint_trajectory(trajectory.sample(0.2), _planar_ik, seed=[0.4, 1.5])
    fine = plan_joint_trajectory(trajectory.sample(0.02), _planar_ik, seed=[0.4, 1.5])
    assert coarse.ok and fine.ok
    assert fine.max_joint_step < coarse.max_joint_step
