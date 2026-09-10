"""Rigid-body model of a standard six-sided die, and a re-orientation planner.

This module is deliberately free of any ROS dependency so that the whole
re-orientation logic can be unit-tested offline (see ``tests/test_die_model.py``).

Face convention
---------------
The face normals below match the ones published by ``drims_dice_simulator``
(``dice_spawner.py``), expressed in the die *body* frame::

    1 -> -Z      6 -> +Z
    2 -> -X      5 -> +X
    3 -> +Y      4 -> -Y

which satisfies the standard die invariant ``face + opposite(face) == 7``.

Orientation representation
--------------------------
The orientation of the die is a rotation matrix ``R`` mapping *body* coordinates
into *world* (robot base) coordinates.  Because the die always rests flat on the
board, ``R`` is always an element of the 24-element rotation group of the cube,
so every matrix has integer entries in ``{-1, 0, 1}``.

Robot primitive
---------------
The only way a parallel-jaw gripper approaching from above can change which face
points up is:

1. close the fingers on two *opposite lateral* faces of the die,
2. lift,
3. rotate the wrist by +/-90 degrees about the horizontal axis that passes
   through the two grasped faces,
4. put the die back down and release.

That primitive is modelled by :class:`Regrasp`.  The grasp axis is horizontal and
is chosen by the planner, so the reachable rotations are +/-90 degrees about the
world X or Y axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FACE_NORMALS",
    "opposite",
    "Regrasp",
    "PRIMITIVES",
    "identity_orientation",
    "orientation_from_up_and_yaw",
    "orientation_from_two_faces",
    "up_face",
    "face_towards",
    "apply",
    "plan_exact",
    "plan_from_normals",
    "nearest_cube_rotation",
    "CUBE_ROTATIONS",
    "next_blind_move",
    "DeductivePolicy",
    "BlindSearchPolicy",
]

# --------------------------------------------------------------------------- #
# Die geometry
# --------------------------------------------------------------------------- #

FACE_NORMALS: Dict[int, np.ndarray] = {
    1: np.array([0.0, 0.0, -1.0]),
    2: np.array([-1.0, 0.0, 0.0]),
    3: np.array([0.0, 1.0, 0.0]),
    4: np.array([0.0, -1.0, 0.0]),
    5: np.array([1.0, 0.0, 0.0]),
    6: np.array([0.0, 0.0, 1.0]),
}

_WORLD_UP = np.array([0.0, 0.0, 1.0])


def opposite(face: int) -> int:
    """Return the face on the other side of the die (``face + result == 7``)."""
    if not 1 <= face <= 6:
        raise ValueError(f"face must be in 1..6, got {face}")
    return 7 - face


# --------------------------------------------------------------------------- #
# Rotations
# --------------------------------------------------------------------------- #


def _rot(axis: str, quarter_turns: int) -> np.ndarray:
    """Rotation matrix of ``quarter_turns * 90`` degrees about a world axis."""
    theta = 0.5 * np.pi * quarter_turns
    c, s = round(np.cos(theta)), round(np.sin(theta))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(f"unknown axis {axis!r}")


@dataclass(frozen=True)
class Regrasp:
    """One grasp-lift-rotate-place cycle.

    Attributes
    ----------
    axis:
        Horizontal world axis the wrist rotates about, ``"x"`` or ``"y"``.  This
        is also the direction along which the fingers close, because the fingers
        must sit on the two faces that the rotation leaves in place.
    quarter_turns:
        ``+1`` or ``-1``; the sign selects the rotation direction.  Both are
        kinematically valid, and the executor picks whichever keeps the die
        inside the workspace.
    """

    axis: str
    quarter_turns: int

    def matrix(self) -> np.ndarray:
        return _rot(self.axis, self.quarter_turns)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        sign = "+" if self.quarter_turns > 0 else "-"
        return f"rotate {sign}90deg about {self.axis.upper()}"


#: The four primitives available to the planner.
PRIMITIVES: Tuple[Regrasp, ...] = tuple(
    Regrasp(axis, turns) for axis, turns in product(("x", "y"), (1, -1))
)


def _generate_cube_rotations() -> Tuple[np.ndarray, ...]:
    """The 24 orientations of a cube, generated from the quarter turns."""
    def key(mat: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(round(v)) for v in mat.flatten())

    found = {key(np.eye(3)): np.eye(3)}
    frontier = [np.eye(3)]
    while frontier:
        current = frontier.pop()
        for axis in ("x", "y", "z"):
            nxt = _rot(axis, 1) @ current
            if key(nxt) not in found:
                found[key(nxt)] = nxt
                frontier.append(nxt)
    return tuple(found.values())


#: Every orientation a die can rest in on a flat board.
CUBE_ROTATIONS: Tuple[np.ndarray, ...] = _generate_cube_rotations()


def nearest_cube_rotation(rotation: np.ndarray) -> np.ndarray:
    """Snap an arbitrary rotation to the closest of the 24 cube orientations.

    Measured face normals never land exactly on the lattice: TF carries noise,
    and a die knocked askew or a mis-estimated yaw can be tens of degrees out.
    Snapping keeps the planner well defined instead of searching a lattice the
    measurement is not on -- which would otherwise fail to find any plan at all.
    """
    rotation = np.asarray(rotation, dtype=float)
    return max(CUBE_ROTATIONS, key=lambda candidate: float(np.trace(candidate.T @ rotation)))


def identity_orientation() -> np.ndarray:
    """Die sitting with face 6 up and face 5 pointing along +X."""
    return np.eye(3)


def orientation_from_up_and_yaw(up: int, yaw_quarter_turns: int = 0) -> np.ndarray:
    """Build an orientation with ``up`` pointing at +Z.

    ``yaw_quarter_turns`` rotates the die about the world Z axis afterwards,
    which changes nothing about the up face but does change which lateral face
    is where.  Useful for exhaustive tests.
    """
    n = FACE_NORMALS[up]
    # Find any rotation taking n onto +Z.
    if np.allclose(n, _WORLD_UP):
        base = np.eye(3)
    elif np.allclose(n, -_WORLD_UP):
        base = _rot("x", 2)
    else:
        axis = np.cross(n, _WORLD_UP)
        axis = axis / np.linalg.norm(axis)
        # n is a unit axis vector, so the angle is always 90 degrees here.
        if abs(axis[0]) > 0.5:
            base = _rot("x", 1 if np.dot(np.cross(n, _WORLD_UP), [1, 0, 0]) > 0 else -1)
        else:
            base = _rot("y", 1 if np.dot(np.cross(n, _WORLD_UP), [0, 1, 0]) > 0 else -1)
    return _rot("z", yaw_quarter_turns) @ base


def face_towards(orientation: np.ndarray, direction: Sequence[float]) -> int:
    """Which face of the die points along ``direction`` for this orientation."""
    d = np.asarray(direction, dtype=float)
    best_face, best_dot = 0, -np.inf
    for face, normal in FACE_NORMALS.items():
        dot = float(np.dot(orientation @ normal, d))
        if dot > best_dot:
            best_dot, best_face = dot, face
    return best_face


def up_face(orientation: np.ndarray) -> int:
    """Which face of the die points at +Z for the given orientation."""
    return face_towards(orientation, _WORLD_UP)


def orientation_from_two_faces(
    up_value: int, side_value: int, side_direction: Sequence[float]
) -> Optional[np.ndarray]:
    """Recover the die's full orientation from two observed faces.

    A die is *chiral*: 1, 2 and 3 run anticlockwise about their shared vertex,
    and no rotation can turn a die into its mirror image.  So the 24 orientations
    produce 24 distinct ``(top face, one lateral face)`` pairs, and knowing the
    top face plus **one** lateral face pins the orientation down completely --
    there is nothing left to guess about the other three faces.

    That is the whole reason the policy below only ever needs a single probe
    turn.  Returns ``None`` if no orientation matches, which means the two
    observations are mutually inconsistent (a misread face, or the die was
    nudged between them) rather than merely incomplete.
    """
    if not 1 <= up_value <= 6 or not 1 <= side_value <= 6:
        raise ValueError("face values must be in 1..6")
    for candidate in CUBE_ROTATIONS:
        if up_face(candidate) == up_value and face_towards(candidate, side_direction) == side_value:
            return candidate
    return None


def apply(orientation: np.ndarray, move: Regrasp) -> np.ndarray:
    """Rotate the die in the world frame by ``move``."""
    return move.matrix() @ orientation


# --------------------------------------------------------------------------- #
# Exact planner (full orientation known, e.g. from the simulator's face TFs)
# --------------------------------------------------------------------------- #


def plan_exact(orientation: np.ndarray, target_face: int) -> List[Regrasp]:
    """Shortest sequence of re-grasps bringing ``target_face`` to the top.

    Breadth-first search over the four primitives.  Because a quarter turn about
    X or Y can reach any of the four lateral faces, and two turns reach the
    bottom face, the answer is never longer than two moves -- but the search is
    written generically so that a restricted primitive set (say, a single grasp
    axis) still yields a correct plan.
    """
    if not 1 <= target_face <= 6:
        raise ValueError(f"target_face must be in 1..6, got {target_face}")

    if up_face(orientation) == target_face:
        return []

    # Key states by their integer matrix so revisits are cheap to detect.
    def key(mat: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(round(v)) for v in mat.flatten())

    frontier: List[Tuple[np.ndarray, List[Regrasp]]] = [(orientation, [])]
    seen = {key(orientation)}

    while frontier:
        current, path = frontier.pop(0)
        for move in PRIMITIVES:
            nxt = apply(current, move)
            k = key(nxt)
            if k in seen:
                continue
            new_path = path + [move]
            if up_face(nxt) == target_face:
                return new_path
            seen.add(k)
            frontier.append((nxt, new_path))

    raise RuntimeError(  # pragma: no cover - unreachable with PRIMITIVES
        f"no plan found to bring face {target_face} up"
    )


def plan_from_normals(
    face_normals_world: Dict[int, Sequence[float]], target_face: int
) -> List[Regrasp]:
    """Plan from measured face normals instead of a rotation matrix.

    ``face_normals_world`` maps each face id to that face's outward normal
    expressed in the world/base frame -- exactly what you get by looking up the
    ``face{1..6}_tf`` frames published by ``drims_dice_simulator`` and rotating
    their local +Z into the base frame.  The nearest cube-group orientation is
    recovered by orthonormalising the measurement, which makes the planner
    robust to the small numerical noise in TF.
    """
    # Columns of R are the world directions of the body axes X, Y, Z.
    # Body +X is face 5's normal, +Y is face 3's, +Z is face 6's.
    cols = []
    for face in (5, 3, 6):
        if face in face_normals_world:
            cols.append(np.asarray(face_normals_world[face], dtype=float))
        else:
            cols.append(-np.asarray(face_normals_world[opposite(face)], dtype=float))
    raw = np.column_stack(cols)

    # Project onto SO(3) (nearest rotation matrix in the Frobenius sense), then
    # snap to integers because the die rests flat on the board.
    u, _, vt = np.linalg.svd(raw)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return plan_exact(nearest_cube_rotation(rot), target_face)


# --------------------------------------------------------------------------- #
# Blind planner (only the up face is observable, e.g. a single overhead camera)
# --------------------------------------------------------------------------- #


class DeductivePolicy:
    """Closed-loop policy when only the *top* face can be observed.

    An overhead camera reads the number on the top face and nothing else, so the
    die's full orientation is unknown.  The policy is the three-stage reasoning
    a person uses on a real die:

    1. **Check the top.**  If it already shows the target, leave the die alone.
    2. **The target may be underneath.**  Opposite faces sum to seven, so the
       bottom face is known the moment the top is read.  If the target is the
       bottom face, two quarter turns about the same horizontal axis bring it
       up -- 180 degrees, the shortest route from bottom to top.
    3. **Otherwise the target is one of the four sides.**  Turn the die once by
       90 degrees to bring a side face into view.  That single probe is enough:
       a die is chiral, so *top plus one lateral face determines the whole
       configuration* (:func:`orientation_from_two_faces`).  From there the
       robot knows exactly where the target is and drives straight to it with
       :func:`plan_exact`.

    Stages 2 and 3 are the same instruction to the robot -- "turn 90 degrees
    about the current grasp axis" -- so the implementation issues one probe and
    then always knows everything.  If the target happened to be the bottom, the
    probe reveals a side, the deduction confirms the target is now the new
    bottom, and one more quarter turn in the same direction finishes it: exactly
    the "rotate twice in the same axis" of stage 2.

    Cost, uniformly over all 24 orientations and all 6 targets:

    ==========================  ======  =============
    where the target starts     cases   re-grasps
    ==========================  ======  =============
    already on top              1 / 6   0
    the probed side             1 / 6   1
    on the grasp axis           2 / 6   2
    the bottom                  1 / 6   2
    opposite the probed side    1 / 6   3
    ==========================  ======  =============

    Mean 5/3 = 1.67 re-grasps, worst case 3.  The previous cycle-walking policy
    averaged 2.0 with a worst case of 4; deducing the configuration after one
    probe removes a third of the arm motion.  It cannot be beaten without seeing
    more of the die, because the four sides are indistinguishable until one of
    them is turned up, and any first probe leaves them at 1, 2, 2 and 3 turns.
    With the full pose available -- the simulator publishes ``face{1..6}_tf`` --
    :func:`plan_from_normals` does better still: mean 1.0, worst case 2.
    """

    def __init__(self, target_face: int, first_axis: str = "x") -> None:
        if not 1 <= target_face <= 6:
            raise ValueError(f"target_face must be in 1..6, got {target_face}")
        if first_axis not in ("x", "y"):
            raise ValueError(f"first_axis must be 'x' or 'y', got {first_axis!r}")
        self.target_face = target_face
        self.axis = first_axis
        #: The deduced orientation, or ``None`` while the die is still unknown.
        self.orientation: Optional[np.ndarray] = None
        #: One-line account of the last deduction, for logging.
        self.reasoning: str = "nothing observed yet"
        self._probe: Optional[Regrasp] = None
        self._top_before_probe: Optional[int] = None

    # -- knowledge --------------------------------------------------------- #

    def _deduce(self, observed_face: int) -> None:
        """Turn (top before probe, top after probe) into a full orientation."""
        probe, before_top = self._probe, self._top_before_probe
        self._probe = self._top_before_probe = None
        if probe is None or before_top is None:
            return

        # The probe lifted whichever face pointed along ``direction`` onto +Z,
        # so ``observed_face`` was the face in that direction beforehand.
        direction = probe.matrix().T @ _WORLD_UP
        before = orientation_from_two_faces(before_top, observed_face, direction)
        if before is None:
            self.reasoning = (
                f"saw {before_top} then {observed_face} after {probe}, which no "
                "die orientation explains; probing again"
            )
            return

        self.orientation = apply(before, probe)
        lateral = ", ".join(
            f"{name}={face_towards(self.orientation, vector)}"
            for name, vector in (
                ("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
                ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
            )
        )
        self.reasoning = (
            f"top was {before_top}, {probe.axis.upper()}-turn showed "
            f"{observed_face}: configuration is top={up_face(self.orientation)}, "
            f"bottom={opposite(up_face(self.orientation))}, {lateral}"
        )

    # -- the policy -------------------------------------------------------- #

    def observe(self, current_face: int) -> Optional[Regrasp]:
        """Report the observed top face; get the next move, or ``None`` if done."""
        if not 1 <= current_face <= 6:
            raise ValueError(f"observed face must be in 1..6, got {current_face}")

        if self._probe is not None:
            self._deduce(current_face)

        if current_face == self.target_face:
            self.reasoning = f"face {current_face} is already up"
            return None

        # A model that disagrees with the camera is worse than no model: the die
        # was nudged, or a face was misread.  Drop it and probe again.
        if self.orientation is not None and up_face(self.orientation) != current_face:
            self.reasoning = (
                f"expected {up_face(self.orientation)} up but saw {current_face}; "
                "the die moved, re-deducing"
            )
            self.orientation = None

        if self.orientation is not None:
            move = plan_exact(self.orientation, self.target_face)[0]
            self.orientation = apply(self.orientation, move)
            return move

        # Nothing known beyond the top and bottom faces.  One quarter turn about
        # the grasp axis both makes progress and reveals a side face.
        self._probe = Regrasp(self.axis, 1)
        self._top_before_probe = current_face
        if self.target_face == opposite(current_face):
            self.reasoning = (
                f"target {self.target_face} is the bottom face (7 - {current_face}); "
                "turning 180 degrees about the same axis, one quarter at a time"
            )
        else:
            self.reasoning = (
                f"target {self.target_face} is on a side; turning once about "
                f"{self.axis.upper()} to read a side face and pin the configuration down"
            )
        return self._probe


#: Kept so older configs and notes that say "blind" keep working.
BlindSearchPolicy = DeductivePolicy


def next_blind_move(
    current_face: int, target_face: int, axis: str = "x"
) -> Optional[Regrasp]:
    """One-shot convenience wrapper around :class:`DeductivePolicy`."""
    return DeductivePolicy(target_face, axis).observe(current_face)


def simulate_blind(
    orientation: np.ndarray, target_face: int, first_axis: str = "x", max_moves: int = 12
) -> List[Regrasp]:
    """Run :class:`DeductivePolicy` against a known orientation.

    Used by the test-suite to bound the worst case; the robot itself never has
    access to ``orientation`` -- it only ever sees ``up_face(state)``.
    """
    policy = DeductivePolicy(target_face, first_axis)
    moves: List[Regrasp] = []
    state = orientation
    for _ in range(max_moves):
        move = policy.observe(up_face(state))
        if move is None:
            return moves
        state = apply(state, move)
        moves.append(move)
    raise RuntimeError(
        f"deductive policy did not converge in {max_moves} moves "
        f"(target {target_face}, first axis {first_axis})"
    )


def describe_plan(moves: Iterable[Regrasp]) -> str:
    """Human-readable one-liner for logging."""
    moves = list(moves)
    if not moves:
        return "already showing the requested face"
    return " then ".join(str(m) for m in moves)
