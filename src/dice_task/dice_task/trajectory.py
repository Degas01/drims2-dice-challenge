"""Cartesian path and trajectory generation, following the DRIMS motion-planning lecture.

The lecture splits the problem in two, and so does this module:

**Path generation** -- "geometrically define a curve from point A to point B",
using geometrical primitives.  :class:`LinearPath` is the LIN primitive:
``p(s) = p_a + s (p_b - p_a) / L`` for ``s`` in ``[0, L]``.

**Trajectory generation** -- "find a motion profile *along the path*", where
time matters.  :class:`TrapezoidalProfile` is the minimum-time profile subject
to ``|s'| <= v_max`` and ``|s''| <= a_max``, i.e. the

    ``s(t) = argmin ∫dt``  subject to  ``s(0)=0, s(T)=L, |s'|<=v_max, |s''|<=a_max``

problem from the slides.  It degenerates to the triangular profile when the path
is too short to ever reach ``v_max`` -- the case that is easy to forget and that
makes short approach moves jerk.

**Orientation** is interpolated with SLERP, as the lecture presents it:

    ``Q(t) = Q_a * (Q_a^-1 * Q_b)^s(t)``

and translation and rotation are **synchronised in time**, so the pose reaches
its target position and orientation together.  That matters for a grasp: a wrist
that finishes turning after the fingers arrive will clip the die.

**Blending** between consecutive segments uses the blend radius of the slides:
inside a sphere of radius ``r`` around a waypoint the robot is allowed to cut
the corner rather than stop.  Removing those full stops is the single biggest
cycle-time saving available in this task, and cycle time is precisely what the
challenge scores.

Everything here is ROS-free and unit-tested in ``tests/test_trajectory.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TrapezoidalProfile",
    "LinearPath",
    "slerp",
    "quaternion_angle",
    "CartesianSegment",
    "CartesianTrajectory",
    "CartesianPath",
    "BlendedTrajectory",
    "blend_waypoints",
]


# --------------------------------------------------------------------------- #
# Motion profile along the path
# --------------------------------------------------------------------------- #


@dataclass
class TrapezoidalProfile:
    """Minimum-time rest-to-rest profile for a path of length ``length``.

    Attributes
    ----------
    length:
        Arc length to traverse.
    v_max, a_max:
        Velocity and acceleration limits, in the same units as ``length``.
    """

    length: float
    v_max: float
    a_max: float

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError(f"length must be non-negative, got {self.length}")
        if self.v_max <= 0 or self.a_max <= 0:
            raise ValueError("v_max and a_max must be positive")

        if self.length == 0.0:
            self.v_peak = 0.0
            self.t_acc = 0.0
            self.t_flat = 0.0
            self.duration = 0.0
            return

        # Distance needed to accelerate to v_max and back down again.
        ramp_distance = self.v_max**2 / self.a_max
        if self.length >= ramp_distance:
            # Full trapezoid: accelerate, cruise, decelerate.
            self.v_peak = self.v_max
            self.t_acc = self.v_max / self.a_max
            self.t_flat = (self.length - ramp_distance) / self.v_max
        else:
            # Triangular: the path is too short to ever reach v_max.
            self.t_acc = float(np.sqrt(self.length / self.a_max))
            self.v_peak = self.a_max * self.t_acc
            self.t_flat = 0.0
        self.duration = 2.0 * self.t_acc + self.t_flat

    # -- evaluation -------------------------------------------------------- #

    def position(self, t: float) -> float:
        """Arc length covered by time ``t``."""
        if self.duration == 0.0:
            return 0.0
        t = float(np.clip(t, 0.0, self.duration))
        if t <= self.t_acc:
            return 0.5 * self.a_max * t * t
        if t <= self.t_acc + self.t_flat:
            return 0.5 * self.a_max * self.t_acc**2 + self.v_peak * (t - self.t_acc)
        # Decelerating: mirror the acceleration phase about the end.
        remaining = self.duration - t
        return self.length - 0.5 * self.a_max * remaining * remaining

    def velocity(self, t: float) -> float:
        if self.duration == 0.0:
            return 0.0
        t = float(np.clip(t, 0.0, self.duration))
        if t <= self.t_acc:
            return self.a_max * t
        if t <= self.t_acc + self.t_flat:
            return self.v_peak
        return self.a_max * (self.duration - t)

    def acceleration(self, t: float) -> float:
        if self.duration == 0.0:
            return 0.0
        if t < self.t_acc:
            return self.a_max
        if t <= self.t_acc + self.t_flat:
            return 0.0
        if t <= self.duration:
            return -self.a_max
        return 0.0

    def normalised(self, t: float) -> float:
        """Progress in ``[0, 1]``; this is the ``s(t)`` used by SLERP."""
        if self.length == 0.0:
            return 1.0
        return self.position(t) / self.length

    def stretched_to(self, duration: float) -> "TrapezoidalProfile":
        """Same path, slowed down so it takes exactly ``duration`` seconds.

        Used to synchronise translation and rotation: the slower of the two sets
        the pace and the faster one is stretched to match, so both finish
        together instead of the wrist still turning after the tool has arrived.
        """
        if duration <= 0 or self.duration == 0.0:
            return self
        if duration < self.duration - 1e-12:
            raise ValueError(
                f"cannot stretch a {self.duration:.4f}s profile to {duration:.4f}s; "
                "that would violate the limits"
            )
        scale = self.duration / duration
        return TrapezoidalProfile(
            self.length, self.v_max * scale, self.a_max * scale * scale
        )


# --------------------------------------------------------------------------- #
# Path primitives
# --------------------------------------------------------------------------- #


@dataclass
class LinearPath:
    """The LIN primitive: a straight segment in Cartesian space."""

    start: np.ndarray
    end: np.ndarray

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=float).reshape(3)
        self.end = np.asarray(self.end, dtype=float).reshape(3)
        self.delta = self.end - self.start
        self.length = float(np.linalg.norm(self.delta))
        self.direction = (
            self.delta / self.length if self.length > 1e-12 else np.zeros(3)
        )

    def at_arclength(self, s: float) -> np.ndarray:
        """Point at arc length ``s`` from the start."""
        if self.length <= 1e-12:
            return self.start.copy()
        s = float(np.clip(s, 0.0, self.length))
        return self.start + s * self.direction

    def at(self, u: float) -> np.ndarray:
        """Point at normalised parameter ``u`` in ``[0, 1]``."""
        return self.at_arclength(float(np.clip(u, 0.0, 1.0)) * self.length)


def slerp(q_a: Sequence[float], q_b: Sequence[float], u: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions ``[x, y, z, w]``.

    Implements ``Q(u) = Q_a (Q_a^-1 Q_b)^u`` from the lecture, taking the short
    way round: a quaternion and its negation are the same rotation, so the sign
    of ``q_b`` is flipped when the pair is more than 90 degrees apart in
    four-space.  Without that a 170-degree turn becomes a 190-degree turn in the
    opposite direction, which is how a wrist ends up unwinding into a joint
    limit halfway through a grasp.
    """
    a = np.asarray(q_a, dtype=float).reshape(4)
    b = np.asarray(q_b, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    u = float(np.clip(u, 0.0, 1.0))
    if dot > 0.9995:
        # Nearly identical: linear interpolation is numerically better here.
        result = a + u * (b - a)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - u) * theta) / sin_theta) * a + (
        np.sin(u * theta) / sin_theta
    ) * b


def quaternion_angle(q_a: Sequence[float], q_b: Sequence[float]) -> float:
    """Rotation angle between two orientations, in radians, in ``[0, pi]``."""
    a = np.asarray(q_a, dtype=float).reshape(4)
    b = np.asarray(q_b, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = abs(float(np.dot(a, b)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Cartesian trajectory: LIN + SLERP, synchronised
# --------------------------------------------------------------------------- #


@dataclass
class CartesianSegment:
    """One LIN move with a synchronised orientation change."""

    start_position: np.ndarray
    end_position: np.ndarray
    start_quaternion: np.ndarray
    end_quaternion: np.ndarray

    v_max: float = 0.25  # m/s
    a_max: float = 0.5  # m/s^2
    w_max: float = 1.5  # rad/s
    alpha_max: float = 3.0  # rad/s^2

    def __post_init__(self) -> None:
        self.path = LinearPath(self.start_position, self.end_position)
        self.start_quaternion = np.asarray(self.start_quaternion, float).reshape(4)
        self.end_quaternion = np.asarray(self.end_quaternion, float).reshape(4)
        self.angle = quaternion_angle(self.start_quaternion, self.end_quaternion)

        translation = TrapezoidalProfile(self.path.length, self.v_max, self.a_max)
        rotation = TrapezoidalProfile(self.angle, self.w_max, self.alpha_max)

        # Whichever of the two takes longer sets the pace; the other is
        # stretched so both finish together.
        self.duration = max(translation.duration, rotation.duration)
        self.translation = translation.stretched_to(self.duration)
        self.rotation = rotation.stretched_to(self.duration)

    def pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Position and orientation at time ``t`` from the segment's start."""
        position = self.path.at_arclength(self.translation.position(t))
        orientation = slerp(
            self.start_quaternion, self.end_quaternion, self.rotation.normalised(t)
        )
        return position, orientation

    def twist_at(self, t: float) -> Tuple[float, float]:
        """Linear and angular speed at time ``t``."""
        return self.translation.velocity(t), self.rotation.velocity(t)


class CartesianTrajectory:
    """A sequence of :class:`CartesianSegment` traversed back to back."""

    def __init__(self, segments: Sequence[CartesianSegment]) -> None:
        self.segments = list(segments)
        self.starts: List[float] = []
        total = 0.0
        for segment in self.segments:
            self.starts.append(total)
            total += segment.duration
        self.duration = total

    @classmethod
    def through(
        cls,
        waypoints: Sequence[Tuple[Sequence[float], Sequence[float]]],
        **limits,
    ) -> "CartesianTrajectory":
        """Build a trajectory through ``(position, quaternion)`` waypoints."""
        if len(waypoints) < 2:
            raise ValueError("need at least two waypoints")
        segments = [
            CartesianSegment(
                np.asarray(waypoints[i][0], float),
                np.asarray(waypoints[i + 1][0], float),
                np.asarray(waypoints[i][1], float),
                np.asarray(waypoints[i + 1][1], float),
                **limits,
            )
            for i in range(len(waypoints) - 1)
        ]
        return cls(segments)

    def pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.segments:
            raise ValueError("empty trajectory")
        t = float(np.clip(t, 0.0, self.duration))
        for start, segment in zip(reversed(self.starts), reversed(self.segments)):
            if t >= start - 1e-12:
                return segment.pose_at(t - start)
        return self.segments[0].pose_at(0.0)

    def sample(self, dt: float) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """Sample the trajectory every ``dt`` seconds, endpoint included."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        times = list(np.arange(0.0, self.duration, dt))
        if not times or times[-1] < self.duration - 1e-9:
            times.append(self.duration)
        return [(float(t), *self.pose_at(t)) for t in times]

    def path_length(self) -> float:
        return float(sum(segment.path.length for segment in self.segments))


# --------------------------------------------------------------------------- #
# Corner blending
# --------------------------------------------------------------------------- #


def blend_waypoints(
    waypoints: Sequence[Sequence[float]], blend_radius: float
) -> List[np.ndarray]:
    """Round off the corners of a polyline, as the blend radius does.

    Each interior waypoint is replaced by a quadratic Bezier arc that leaves the
    incoming segment ``blend_radius`` before the corner and rejoins the outgoing
    one ``blend_radius`` after it, so the robot never has to stop at the corner.
    The radius is clipped to half of the shorter adjacent segment, which is what
    keeps two nearby corners from overlapping into a shortcut that misses the
    waypoints entirely.

    Returns a denser polyline; feed it to :meth:`CartesianTrajectory.through`.
    """
    points = [np.asarray(p, dtype=float).reshape(3) for p in waypoints]
    if len(points) < 3 or blend_radius <= 0:
        return points

    blended: List[np.ndarray] = [points[0]]
    for i in range(1, len(points) - 1):
        previous, corner, following = points[i - 1], points[i], points[i + 1]
        into = corner - previous
        out_of = following - corner
        len_in, len_out = np.linalg.norm(into), np.linalg.norm(out_of)
        if len_in < 1e-9 or len_out < 1e-9:
            blended.append(corner)
            continue

        radius = min(blend_radius, 0.5 * len_in, 0.5 * len_out)
        enter = corner - radius * into / len_in
        leave = corner + radius * out_of / len_out

        blended.append(enter)
        # Quadratic Bezier through the corner: a parabolic blend, which is the
        # "circular/parabolic blend" the lecture names.  Sampled finely enough
        # that no residual vertex counts as a hard corner (see hard_corners).
        for step in np.linspace(0.0, 1.0, 11)[1:-1]:
            one_minus = 1.0 - step
            blended.append(
                one_minus * one_minus * enter
                + 2.0 * one_minus * step * corner
                + step * step * leave
            )
        blended.append(leave)

    blended.append(points[-1])
    return blended


# --------------------------------------------------------------------------- #
# Continuous traversal of a blended path
# --------------------------------------------------------------------------- #


class CartesianPath:
    """A polyline in position with piecewise-SLERP orientation, by arc length.

    Purely geometric: no time, no limits.  This is the "path generation" half of
    the lecture, kept separate from the "trajectory generation" half so that the
    same geometry can be traversed with different motion profiles.
    """

    def __init__(
        self, waypoints: Sequence[Tuple[Sequence[float], Sequence[float]]]
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("need at least two waypoints")
        self.positions = [np.asarray(p, float).reshape(3) for p, _ in waypoints]
        self.quaternions = [np.asarray(q, float).reshape(4) for _, q in waypoints]

        self.cumulative = [0.0]
        for a, b in zip(self.positions, self.positions[1:]):
            self.cumulative.append(self.cumulative[-1] + float(np.linalg.norm(b - a)))
        self.length = self.cumulative[-1]

    def pose_at_arclength(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        s = float(np.clip(s, 0.0, self.length))
        if self.length <= 1e-12:
            return self.positions[0].copy(), self.quaternions[0].copy()

        index = int(np.searchsorted(self.cumulative, s, side="right")) - 1
        index = int(np.clip(index, 0, len(self.positions) - 2))
        span = self.cumulative[index + 1] - self.cumulative[index]
        u = 0.0 if span <= 1e-12 else (s - self.cumulative[index]) / span

        position = self.positions[index] + u * (
            self.positions[index + 1] - self.positions[index]
        )
        orientation = slerp(self.quaternions[index], self.quaternions[index + 1], u)
        return position, orientation

    def total_rotation(self) -> float:
        """Sum of the turn angles between consecutive orientations."""
        return float(
            sum(
                quaternion_angle(a, b)
                for a, b in zip(self.quaternions, self.quaternions[1:])
            )
        )

    def turn_angles(self) -> List[float]:
        """Direction change at each interior vertex, in radians."""
        angles: List[float] = []
        for a, b, c in zip(self.positions, self.positions[1:], self.positions[2:]):
            into, out_of = b - a, c - b
            len_in, len_out = np.linalg.norm(into), np.linalg.norm(out_of)
            if len_in < 1e-12 or len_out < 1e-12:
                angles.append(0.0)
                continue
            cosine = float(np.dot(into / len_in, out_of / len_out))
            angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
        return angles

    def hard_corners(self, max_smooth_turn: float = np.deg2rad(30.0)) -> List[int]:
        """Interior vertices where the tangent direction jumps too much to round.

        A polyline followed *exactly* has zero radius of curvature at every
        vertex -- the tangent is discontinuous -- so strictly no vertex can be
        taken at speed.  What makes a blended path traversable is that the turn
        is spread over many vertices, each small enough that the discrete
        polyline is a good approximation of a smooth arc.  This distinguishes
        the two cases: a turn above the threshold is a genuine corner and forces
        a stop, while the small turns left by :func:`blend_waypoints` do not.
        """
        return [
            index
            for index, angle in enumerate(self.turn_angles(), start=1)
            if angle > max_smooth_turn
        ]

    def min_turn_radius(self, max_smooth_turn: float = np.deg2rad(30.0)) -> float:
        """Radius of curvature of the tightest bend, or 0.0 at a hard corner.

        For the small turns of a blended path the discrete curvature
        ``kappa = theta / ds`` is a faithful estimate, so the radius is
        ``ds / theta`` with ``ds`` the mean of the two adjacent segments.
        """
        if self.hard_corners(max_smooth_turn):
            return 0.0

        radius = np.inf
        angles = self.turn_angles()
        for index, angle in enumerate(angles, start=1):
            if angle < 1e-9:
                continue  # straight
            before = float(np.linalg.norm(self.positions[index] - self.positions[index - 1]))
            after = float(np.linalg.norm(self.positions[index + 1] - self.positions[index]))
            radius = min(radius, 0.5 * (before + after) / angle)
        return float(radius)


class BlendedTrajectory:
    """One motion profile per *smooth* stretch of path -- stops only where needed.

    This is what the blend radius buys.  :class:`CartesianTrajectory` treats
    every waypoint as a full stop, which is what issuing one ``move_to_pose``
    per waypoint gives you: the robot decelerates to zero at each one.  Here the
    path is cut only at *hard corners* -- vertices whose tangent jumps too far
    to be rounded -- and each smooth stretch between them is crossed with a
    single trapezoidal profile.

    Some stops are unavoidable and the model says so.  A pick-and-place reverses
    direction at the grasp: it descends, then climbs back up the same line, a
    180-degree turn that no blend radius can smooth.  The robot has to stop
    there, and it was going to anyway to close the gripper.  The saving comes
    from the *other* corners, which previously also cost a full stop each.

    Where the path is smooth, the speed is additionally capped by lateral
    acceleration, ``v <= sqrt(a_lat R)`` with ``R`` the tightest radius of
    curvature -- the physical reason a tighter blend must be taken slower.
    """

    def __init__(
        self,
        waypoints: Sequence[Tuple[Sequence[float], Sequence[float]]],
        v_max: float = 0.25,
        a_max: float = 0.5,
        w_max: float = 1.5,
        alpha_max: float = 3.0,
        lateral_a_max: Optional[float] = None,
        max_smooth_turn: float = np.deg2rad(30.0),
    ) -> None:
        self.path = CartesianPath(waypoints)
        self.hard_corners = self.path.hard_corners(max_smooth_turn)

        # Cut the path at each hard corner; the corner belongs to both sides, so
        # the pieces share that waypoint and the motion is continuous through it
        # (at zero speed).
        boundaries = [0, *self.hard_corners, len(waypoints) - 1]
        self.pieces: List["_SmoothStretch"] = []
        for begin, end in zip(boundaries, boundaries[1:]):
            if end <= begin:
                continue
            self.pieces.append(
                _SmoothStretch(
                    list(waypoints[begin : end + 1]),
                    v_max=v_max,
                    a_max=a_max,
                    w_max=w_max,
                    alpha_max=alpha_max,
                    lateral_a_max=lateral_a_max,
                )
            )

        self.starts: List[float] = []
        total = 0.0
        for piece in self.pieces:
            self.starts.append(total)
            total += piece.duration
        self.duration = total

    @property
    def v_max(self) -> float:
        """Slowest speed cap across the pieces -- the binding constraint."""
        return min((piece.v_max for piece in self.pieces), default=0.0)

    @property
    def corner_speed_limit(self) -> Optional[float]:
        limits = [
            piece.corner_speed_limit
            for piece in self.pieces
            if piece.corner_speed_limit is not None
        ]
        return min(limits) if limits else None

    @property
    def stop_count(self) -> int:
        """How many times the tool comes to rest, endpoints included."""
        return len(self.pieces) + 1

    def pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.pieces:
            return self.path.pose_at_arclength(0.0)
        t = float(np.clip(t, 0.0, self.duration))
        for start, piece in zip(reversed(self.starts), reversed(self.pieces)):
            if t >= start - 1e-12:
                return piece.pose_at(t - start)
        return self.pieces[0].pose_at(0.0)

    def speed_at(self, t: float) -> float:
        if not self.pieces:
            return 0.0
        t = float(np.clip(t, 0.0, self.duration))
        for start, piece in zip(reversed(self.starts), reversed(self.pieces)):
            if t >= start - 1e-12:
                return piece.profile.velocity(t - start)
        return 0.0

    def sample(self, dt: float) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        times = list(np.arange(0.0, self.duration, dt))
        if not times or times[-1] < self.duration - 1e-9:
            times.append(self.duration)
        return [(float(t), *self.pose_at(t)) for t in times]


class _SmoothStretch:
    """One corner-free run of path, crossed with a single profile."""

    def __init__(
        self,
        waypoints: Sequence[Tuple[Sequence[float], Sequence[float]]],
        v_max: float,
        a_max: float,
        w_max: float,
        alpha_max: float,
        lateral_a_max: Optional[float],
    ) -> None:
        self.path = CartesianPath(waypoints)

        self.v_max = float(v_max)
        self.corner_speed_limit: Optional[float] = None
        if lateral_a_max is not None:
            radius = self.path.min_turn_radius()
            if np.isfinite(radius):
                self.corner_speed_limit = float(np.sqrt(lateral_a_max * radius))
                self.v_max = min(self.v_max, max(self.corner_speed_limit, 1e-6))

        translation = TrapezoidalProfile(self.path.length, self.v_max, a_max)
        rotation = TrapezoidalProfile(self.path.total_rotation(), w_max, alpha_max)
        self.duration = max(translation.duration, rotation.duration)
        self.profile = translation.stretched_to(self.duration)

    def pose_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.path.pose_at_arclength(self.profile.position(t))
