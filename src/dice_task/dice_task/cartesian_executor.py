"""Run a generated Cartesian trajectory through IK and onto the controller.

The motion-planning lecture draws the industrial controller as a pipeline::

    instruction stack -> trajectory generation -> inverse kinematics -> axis controllers
         (>10 Hz)            (>100 Hz)                (>100 Hz)            (>1 kHz)

:mod:`dice_task.trajectory` is the middle box: it turns waypoints into a timed
Cartesian path with a trapezoidal profile and SLERP orientation.  This module is
the next one along.  It samples that path, asks ``easy_motion`` for IK at each
sample, and hands the result to ``execute_trajectory`` as a single timed
``JointTrajectory``.

Doing it this way rather than issuing one ``move_to_pose`` per waypoint is what
makes blending possible at all: MoveIt plans each ``move_to_pose`` to a full
stop, so a five-waypoint pick-and-place pays four decelerations it does not
need.  Sending one trajectory keeps the tool moving through the corners, which
on a realistic re-grasp cycle is worth about a quarter of the cycle time -- and
cycle time is what the challenge actually scores.

Two things this deliberately does **not** do:

* **It does not skip collision checking silently.**  A hand-built joint
  trajectory bypasses MoveIt's planning-scene checks, so the caller has to opt
  in with ``motion_mode: trajectory``, and :func:`plan_joint_trajectory` refuses
  paths whose IK solutions jump -- the signature of an elbow flip that would
  sweep the arm through the table.
* **It does not pretend IK always succeeds.**  A sample the arm cannot reach
  aborts the whole trajectory rather than being quietly dropped, because
  dropping it would silently straighten a curve the planner deliberately bent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["JointWaypoint", "TrajectoryPlan", "plan_joint_trajectory"]


@dataclass
class JointWaypoint:
    """One IK solution, stamped with its time along the trajectory."""

    time: float
    positions: List[float]


@dataclass
class TrajectoryPlan:
    """A time-parameterised joint trajectory, plus why it might be unusable."""

    waypoints: List[JointWaypoint] = field(default_factory=list)
    failure: Optional[str] = None
    max_joint_step: float = 0.0

    @property
    def ok(self) -> bool:
        return self.failure is None and len(self.waypoints) >= 2

    @property
    def duration(self) -> float:
        return self.waypoints[-1].time if self.waypoints else 0.0


def plan_joint_trajectory(
    samples: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    solve_ik,
    seed: Optional[Sequence[float]] = None,
    max_joint_step: float = 0.6,
) -> TrajectoryPlan:
    """Turn timed Cartesian samples into a joint trajectory.

    Parameters
    ----------
    samples:
        ``(time, position, quaternion)`` triples, as produced by
        :meth:`dice_task.trajectory.BlendedTrajectory.sample`.
    solve_ik:
        Callable ``(position, quaternion, seed) -> Optional[list[float]]``.  The
        node passes a thin wrapper around ``easy_motion``'s ``get_ik``; the tests
        pass an analytic stub, which is why this function takes a callable
        instead of a client.
    seed:
        Starting joint configuration.  Each solution seeds the next, which is
        what keeps the arm on one IK branch instead of flipping between
        equally-valid elbow-up and elbow-down solutions mid-path.
    max_joint_step:
        Largest tolerated jump, in radians, between consecutive samples.  A
        larger jump means the solver changed branch: the Cartesian path is still
        smooth but the *joint* path is not, and executing it would throw the arm
        across the cell.  Detected rather than executed.

    Returns
    -------
    A :class:`TrajectoryPlan`; check ``ok`` before executing it.
    """
    plan = TrajectoryPlan()
    if len(samples) < 2:
        plan.failure = "need at least two samples"
        return plan

    # `seed_for_next` is what the solver is given; `previous_solution` is what
    # continuity is measured against. They differ for the very first sample: the
    # seed there is the arm's *current* configuration, which is legitimately far
    # from the start of the path -- the arm still has to travel there. Comparing
    # against it would flag every trajectory as a branch flip.
    seed_for_next: Optional[List[float]] = list(seed) if seed is not None else None
    previous_solution: Optional[List[float]] = None

    for index, (time, position, quaternion) in enumerate(samples):
        solution = solve_ik(position, quaternion, seed_for_next)
        if solution is None or len(solution) == 0:
            plan.failure = (
                f"no IK solution at sample {index} of {len(samples)} "
                f"(t={time:.2f}s, position={np.round(position, 4).tolist()})"
            )
            return plan

        solution = [float(value) for value in solution]
        if previous_solution is not None:
            step = float(
                np.max(np.abs(np.asarray(solution) - np.asarray(previous_solution)))
            )
            plan.max_joint_step = max(plan.max_joint_step, step)
            if step > max_joint_step:
                gap_ms = 1000.0 * (time - samples[index - 1][0])
                plan.failure = (
                    f"IK branch change at sample {index} (t={time:.2f}s): a joint "
                    f"moved {step:.2f} rad in {gap_ms:.0f} ms. "
                    "Executing this would swing the arm through the cell."
                )
                return plan

        plan.waypoints.append(JointWaypoint(time=float(time), positions=solution))
        previous_solution = solution
        seed_for_next = solution

    return plan


def to_joint_trajectory_msg(plan: TrajectoryPlan, joint_names: Sequence[str]):
    """Convert to ``trajectory_msgs/JointTrajectory``.

    Imported lazily so this module stays testable without ROS on the path.
    """
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = list(joint_names)
    for waypoint in plan.waypoints:
        point = JointTrajectoryPoint()
        point.positions = list(waypoint.positions)
        seconds = int(waypoint.time)
        point.time_from_start = Duration(
            sec=seconds, nanosec=int((waypoint.time - seconds) * 1e9)
        )
        message.points.append(point)
    return message


def estimate_joint_velocities(plan: TrajectoryPlan) -> np.ndarray:
    """Finite-difference joint velocities, for checking against limits."""
    if len(plan.waypoints) < 2:
        return np.zeros((0, 0))
    times = np.array([w.time for w in plan.waypoints])
    positions = np.array([w.positions for w in plan.waypoints])
    intervals = np.diff(times)[:, None]
    intervals[intervals <= 0] = np.inf
    return np.diff(positions, axis=0) / intervals
