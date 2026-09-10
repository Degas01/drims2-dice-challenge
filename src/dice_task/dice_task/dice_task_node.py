"""The DRIMS challenge state machine: expose a requested face of the die.

Task 3 of the robot-control hands-on.  The node loops

    identify -> plan -> pick -> lift -> rotate -> place -> release -> identify

until the requested face is showing, then stops.  Every motion goes through
``easy_motion``'s :class:`~easy_motion.motion_client.MotionClient`, as the
hands-on asks; nothing here talks to MoveIt directly.

Two planning strategies
-----------------------
``exact``
    Reads the six ``face{1..6}_tf`` frames, recovers the die's full orientation
    and plans the shortest sequence of re-grasps.  Never more than two, one on
    average.  Those frames come from ``drims_dice_simulator``, so this is the
    simulation strategy.

``deduce``
    Uses only the top face, which is all a single overhead camera can see, and
    closes the loop: check the top, and if it is wrong turn the die once to read
    a side face.  Because a die is chiral, that one reading pins down the whole
    configuration, and the robot then drives straight at the target.  1.67
    re-grasps on average, three in the worst case.  This is the strategy that
    works on the real cell.

``auto`` (the default) uses ``exact`` when the face frames are available and
falls back to ``deduce`` when they are not, so the same node runs in both places.
Both strategies are proved correct offline over all 24 orientations and all six
targets in ``tests/test_die_model.py``.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from easy_motion.motion_client import MotionClient
from easy_motion_msgs.srv import DiceIdentification

from dice_task.die_model import (
    FACE_NORMALS,
    DeductivePolicy,
    Regrasp,
    describe_plan,
    opposite,
    plan_from_normals,
)
from dice_task.cartesian_executor import (
    estimate_joint_velocities,
    plan_joint_trajectory,
    to_joint_trajectory_msg,
)
from dice_task.trajectory import BlendedTrajectory, blend_waypoints
from dice_task.grasping import (
    AXIS_VECTORS,
    flange_position,
    flip_about_approach,
    grasp_orientation,
    nearest_equivalent_grasp,
    normals_in_grasp_frame,
    rotate_about_axis,
    tilt_for_turn,
    yaw_rotation,
)


@dataclass
class Attempt:
    """One re-grasp, recorded so the run can be reported and timed."""

    index: int
    observed_face: int
    move: Optional[Regrasp]
    strategy: str
    seconds: float = 0.0
    succeeded: bool = True


@dataclass
class RunReport:
    target_face: int
    attempts: List[Attempt] = field(default_factory=list)
    final_face: int = 0
    success: bool = False
    seconds: float = 0.0

    def summary(self) -> str:
        moves = sum(1 for a in self.attempts if a.move is not None and a.succeeded)
        verdict = "SUCCESS" if self.success else "FAILED"
        return (
            f"{verdict}: face {self.final_face} up (target {self.target_face}) "
            f"after {moves} re-grasp(s) in {self.seconds:.1f} s"
        )


class VisionClient(Node):
    """Small wrapper around the ``dice_identification`` service."""

    def __init__(self, service_name: str = "dice_identification") -> None:
        super().__init__("dice_task_vision_client", use_global_arguments=False)
        self._client = self.create_client(DiceIdentification, service_name)
        self._service_name = service_name

    def identify(self, timeout: float = 10.0) -> Tuple[int, Optional[PoseStamped], bool]:
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {self._service_name} not available")
        future = self._client.call_async(DiceIdentification.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        result = future.result()
        if result is None:
            self.get_logger().error(
                f"{self._service_name} did not answer within {timeout:.0f} s. "
                "The service exists, so whatever provides it is stuck rather than "
                "absent -- check that node's own log."
            )
            return 0, None, False
        if not result.success:
            self.get_logger().error(
                f"{self._service_name} answered but reported failure "
                f"(face_number={result.face_number}). For the dice simulator this "
                "usually means it could not resolve the die in the planning scene: "
                "try `ros2 service call /reset_dice std_srvs/srv/Trigger \"{}\"` "
                "and check the spawner's terminal."
            )
        return int(result.face_number), result.pose, bool(result.success)


class DiceTaskNode(Node):
    """Owns the parameters, the TF buffer and the state machine."""

    def __init__(self) -> None:
        super().__init__("dice_task")

        self.declare_parameter("target_face", 1)
        self.declare_parameter("strategy", "auto")  # auto | exact | deduce
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("die_frame", "dice_tf")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("tip_frame", "tip")
        self.declare_parameter("identification_service", "dice_identification")

        self.declare_parameter("home_joints", [0.0, -1.97, 2.13, -1.83, -1.50, 0.0])
        self.declare_parameter("gripper_open", 0.045)
        self.declare_parameter("gripper_closed", 0.0)
        self.declare_parameter("gripper_effort", 5.0)
        # Declared with dynamic typing on purpose: YAML 1.1 turns a bare `y`
        # into the boolean True, and a statically typed string parameter would
        # then abort the node inside rclpy with a type error that says nothing
        # about YAML. Accept whatever arrives and normalise it below.
        self.declare_parameter(
            "gripper_close_axis",
            "y",
            ParameterDescriptor(
                dynamic_typing=True,
                description="Tool axis the gripper fingers close along: 'x' or 'y'.",
            ),
        )

        self.declare_parameter("dice_size_m", 0.027)
        # How far the gripper leans off vertical to grasp, in degrees.  45 puts
        # the 90-degree flip symmetrically either side of vertical so neither
        # end of it is horizontal; 0 restores the old straight-down grasp, which
        # cannot put the die back down after a flip.
        self.declare_parameter("grasp_tilt_deg", 45.0)
        # The arm's reach, to its own flange. A UR5e is 850 mm. Used only to
        # turn MoveIt's opaque "-31 / no IK solution" into a message that says
        # which pose was out of range and by how much. Set 0 to disable.
        self.declare_parameter("reach_radius_m", 0.850)
        self.declare_parameter("reach_warn_fraction", 0.92)
        # Ask IK whether a grasp is achievable before committing the arm to it,
        # and try nearby leans and both wrist rolls until one solves. Costs four
        # quick IK calls in the normal case; without it an unreachable
        # orientation is only discovered as MoveIt error -31 mid-cycle.
        self.declare_parameter("verify_ik_before_moving", True)
        # Offsets from grasp_tilt_deg to try, in order of preference.
        self.declare_parameter("grasp_tilt_search_deg", [0.0, -7.0, 7.0, -13.0, 13.0])
        # Flange to fingertips; read from TF when available, this is the fallback.
        self.declare_parameter("tool_length_m", 0.15)
        self.declare_parameter("approach_height_m", 0.10)
        self.declare_parameter("lift_height_m", 0.12)
        self.declare_parameter("place_clearance_m", 0.004)
        self.declare_parameter("velocity_scaling", 0.3)
        self.declare_parameter("approach_velocity_scaling", 0.15)
        self.declare_parameter("max_regrasps", 8)
        self.declare_parameter("settle_seconds", 0.7)
        self.declare_parameter("attached_object_id", "dice")

        # Motion execution: "moveit" issues one move_to_pose per waypoint, which
        # decelerates to a stop at every one. "trajectory" generates the
        # Cartesian path here (LIN + SLERP + trapezoidal profile), blends the
        # corners, solves IK per sample and sends one timed JointTrajectory --
        # the pipeline the planning lecture draws. It is faster but bypasses
        # MoveIt's planning-scene checks, so it is opt-in.
        self.declare_parameter("motion_mode", "moveit")  # moveit | trajectory
        self.declare_parameter("blend_radius_m", 0.03)
        self.declare_parameter("sample_dt", 0.05)
        self.declare_parameter("cartesian_v_max", 0.25)
        self.declare_parameter("cartesian_a_max", 0.5)
        self.declare_parameter("cartesian_w_max", 1.5)
        self.declare_parameter("cartesian_alpha_max", 3.0)
        self.declare_parameter("lateral_a_max", 2.0)
        self.declare_parameter("max_joint_step", 0.6)
        self.declare_parameter(
            "joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tool_length: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def p(self, name: str):
        return self.get_parameter(name).value

    def close_axis(self) -> str:
        """The tool axis the fingers close along, normalised to 'x' or 'y'.

        YAML 1.1 parses a bare ``y`` as the boolean ``True`` (and ``n``, ``no``,
        ``off`` as ``False``), so this parameter can arrive as a bool from an
        unquoted config file. Rather than failing with an opaque type error,
        accept it and say plainly what to fix.
        """
        value = self.p("gripper_close_axis")
        if isinstance(value, bool):
            self.get_logger().warn(
                "gripper_close_axis arrived as a boolean: YAML read the bare "
                f"value as {value}. Quote it in the config file "
                '(gripper_close_axis: "y"). Assuming '
                f"'{'y' if value else 'n'}'."
            )
            value = "y" if value else "n"
        axis = str(value).strip().lower().removeprefix("tool_")
        if axis not in ("x", "y"):
            raise ValueError(
                f"gripper_close_axis must be 'x' or 'y', got {value!r}. "
                "If this came from a YAML file, quote the value."
            )
        return axis

    # ------------------------------------------------------------------ #
    # TF
    # ------------------------------------------------------------------ #

    def wait_for_tf(self, timeout: float = 15.0) -> bool:
        """Block until the die frame is reachable from the base frame.

        The TF buffer only fills while this node is being spun (see ``main``),
        so this doubles as a check that the background executor is actually
        running.  Without it the first lookup fails instantly, the node decides
        the face frames are missing and silently drops to the deductive policy --
        which is a much worse failure than saying so.
        """
        base, die = str(self.p("base_frame")), str(self.p("die_frame"))
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            try:
                self._tf_buffer.lookup_transform(base, die, rclpy.time.Time())
                return True
            except Exception as exc:  # noqa: BLE001 - expected while TF fills
                last_error = str(exc)
            time.sleep(0.2)

        self.get_logger().error(
            f"no transform {base} -> {die} after {timeout:.0f} s: {last_error}"
        )
        self.get_logger().error(
            "Is the cell running (ur5e_N_start.launch.py) and the die spawned "
            "(spawn_dice.launch.py)? Check with: ros2 run tf2_ros tf2_echo "
            f"{base} {die}"
        )
        return False

    def face_normals_in_base(self, timeout: float = 1.0) -> Optional[Dict[int, np.ndarray]]:
        """Outward normals of the six faces, in the base frame.

        ``drims_dice_simulator`` publishes ``face{1..6}_tf`` with each frame's
        local +Z along that face's outward normal, so the normal is simply the
        third column of the frame's rotation.
        """
        base = str(self.p("base_frame"))
        normals: Dict[int, np.ndarray] = {}
        for face in range(1, 7):
            try:
                transform = self._tf_buffer.lookup_transform(
                    base,
                    f"face{face}_tf",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=timeout),
                )
            except Exception:  # noqa: BLE001 - absence is an expected outcome
                return None
            q = transform.transform.rotation
            rot = _matrix_from_quaternion((q.x, q.y, q.z, q.w))
            normals[face] = rot[:, 2]
        return normals

    def tool_length(self) -> float:
        """Distance from the flange to the tool tip, in metres.

        Measured from TF (``tool_frame`` -> ``tip_frame``) so it follows
        whatever gripper is actually mounted, and cached because it cannot
        change while the node is running. Falls back to the parameter if TF is
        not up yet.
        """
        if self._tool_length is not None:
            return self._tool_length

        fallback = float(self.p("tool_length_m"))
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.p("tool_frame")), str(self.p("tip_frame")), rclpy.time.Time()
            )
        except Exception:  # noqa: BLE001 - fall back quietly, this is a refinement
            return fallback

        t = transform.transform.translation
        self._tool_length = float(math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z))
        self.get_logger().info(
            f"tool length {self._tool_length * 1000:.0f} mm "
            f"({self.p('tool_frame')} -> {self.p('tip_frame')})"
        )
        return self._tool_length

    def lookup_tool_orientation(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Current tool orientation as ``[x, y, z, w]``, or ``None``.

        Only used to break the tie between two equally valid grasps, so a
        failed lookup is not worth a warning -- it just means the arm takes
        whichever roll the maths produced first.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.p("base_frame")),
                str(self.p("tip_frame")),
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout),
            )
        except Exception:  # noqa: BLE001 - optional refinement
            return None
        q = transform.transform.rotation
        return np.array([q.x, q.y, q.z, q.w], dtype=float)

    def lookup_die_pose(self, timeout: float = 2.0) -> Optional[Tuple[np.ndarray, float]]:
        """Die centre and yaw in the base frame, from the die TF frame."""
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.p("base_frame")),
                str(self.p("die_frame")),
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cannot look up {self.p('die_frame')}: {exc}")
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        rot = _matrix_from_quaternion((q.x, q.y, q.z, q.w))
        # Yaw of the die about the vertical, taken from its X axis projected
        # onto the board plane, and folded into [-45, 45) because the die is
        # square and only its face directions matter for grasping.
        yaw = float(math.atan2(rot[1, 0], rot[0, 0]))
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        return np.array([t.x, t.y, t.z]), yaw


def _matrix_from_quaternion(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class DiceChallenge:
    """The state machine itself."""

    def __init__(self, node: DiceTaskNode, motion: MotionClient, vision: VisionClient):
        self.node = node
        self.motion = motion
        self.vision = vision
        self.log = node.get_logger()

    # -- primitives -------------------------------------------------------- #

    def _pose(self, position, quaternion) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(self.node.p("base_frame"))
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = float(quaternion[0])
        pose.pose.orientation.y = float(quaternion[1])
        pose.pose.orientation.z = float(quaternion[2])
        pose.pose.orientation.w = float(quaternion[3])
        return pose

    def _move(self, position, quaternion, cartesian: bool, velocity: float) -> bool:
        result = self.motion.move_to_pose(
            self._pose(position, quaternion),
            cartesian_motion=cartesian,
            relative_motion=False,
            velocity_scaling=velocity,
        )
        if result.val != MoveItErrorCodes.SUCCESS:
            self.log.error(f"motion failed with MoveIt error {result.val}")
            return False
        return True

    def go_home(self) -> bool:
        self.log.info("moving to home configuration")
        joints = [float(v) for v in self.node.p("home_joints")]
        result = self.motion.move_to_joint(
            joints, velocity_scaling=float(self.node.p("velocity_scaling"))
        )
        if result.val != MoveItErrorCodes.SUCCESS:
            self.log.error(f"could not reach home: {result.val}")
            return False
        self.open_gripper()
        return True

    def open_gripper(self) -> None:
        self.motion.gripper_command(
            position=float(self.node.p("gripper_open")), max_effort=0.0
        )

    def close_gripper(self) -> None:
        self.motion.gripper_command(
            position=float(self.node.p("gripper_closed")),
            max_effort=float(self.node.p("gripper_effort")),
        )

    # -- the lecture's pipeline: path -> trajectory -> IK -> controller ----- #

    def _solve_ik(self, position, quaternion, seed):
        """Adapter from the executor's callable to easy_motion's get_ik."""
        try:
            result, solution = self.motion.get_ik(
                self._pose(position, quaternion), seed=list(seed) if seed else None
            )
        except Exception as exc:  # noqa: BLE001 - a failed solve is not fatal
            self.log.debug(f"IK call failed: {exc}")
            return None
        if result.val != MoveItErrorCodes.SUCCESS or not solution:
            return None
        return list(solution)

    def run_cartesian_path(
        self, waypoints: List[Tuple[np.ndarray, np.ndarray]], seed=None
    ) -> bool:
        """Blend, time-parameterise, solve IK and execute as one trajectory.

        Falls back to the waypoint-by-waypoint moves if anything about the
        trajectory is not safe to execute -- unreachable samples, or an IK branch
        change. Falling back is the right response: the alternative is either
        refusing to move, or executing a joint path that is discontinuous.
        """
        radius = float(self.node.p("blend_radius_m"))
        positions = [np.asarray(p, float) for p, _ in waypoints]
        orientations = [np.asarray(q, float) for _, q in waypoints]

        blended_positions = blend_waypoints(positions, radius)
        # Re-attach orientations: hold each waypoint's orientation over the
        # points that blending inserted around it, so a corner keeps its wrist
        # angle instead of interpolating through the inserted samples.
        blended: List[Tuple[np.ndarray, np.ndarray]] = []
        for point in blended_positions:
            nearest = int(
                np.argmin([float(np.linalg.norm(point - p)) for p in positions])
            )
            blended.append((point, orientations[nearest]))

        trajectory = BlendedTrajectory(
            blended,
            v_max=float(self.node.p("cartesian_v_max")),
            a_max=float(self.node.p("cartesian_a_max")),
            w_max=float(self.node.p("cartesian_w_max")),
            alpha_max=float(self.node.p("cartesian_alpha_max")),
            lateral_a_max=float(self.node.p("lateral_a_max")),
        )
        samples = trajectory.sample(float(self.node.p("sample_dt")))

        plan = plan_joint_trajectory(
            samples,
            self._solve_ik,
            seed=seed,
            max_joint_step=float(self.node.p("max_joint_step")),
        )
        if not plan.ok:
            self.log.warn(f"trajectory rejected: {plan.failure}")
            self.log.warn("falling back to waypoint-by-waypoint motion")
            return False

        velocities = estimate_joint_velocities(plan)
        self.log.info(
            f"executing {len(plan.waypoints)} points over {plan.duration:.2f} s "
            f"({trajectory.stop_count} stops, peak joint rate "
            f"{float(np.max(np.abs(velocities))):.2f} rad/s)"
        )
        message = to_joint_trajectory_msg(plan, list(self.node.p("joint_names")))
        result = self.motion.execute_trajectory(message)
        if result is not None and getattr(result, "val", 0) != MoveItErrorCodes.SUCCESS:
            self.log.error(f"trajectory execution failed: {result.val}")
            return False
        return True

    # -- reachability ------------------------------------------------------ #

    def warn_if_near_reach_limit(self, label: str, position, quaternion) -> bool:
        """Say so, loudly, when a pose is close to the arm's reach.

        MoveIt reports an unreachable pose as ``-31`` ("no IK solution"), which
        is true but says nothing about *why* -- and the arm has usually not
        moved, so there is nothing to look at either. Reach is the most common
        cause on this cell, because the die can sit most of a metre from the
        base and the gripper adds another 150 mm beyond the flange, so it is
        worth measuring explicitly and naming.

        The number that matters is the distance to the **flange**, not to the
        fingertips: the arm's reach is quoted to its own wrist, and everything
        past that is payload. That is also why gripper orientation changes
        reachability at a fixed grasp point -- leaning the tool swings the
        flange through several centimetres.
        """
        reach = float(self.node.p("reach_radius_m"))
        if reach <= 0.0:
            return True

        flange = flange_position(position, quaternion, self.node.tool_length())
        distance = float(np.linalg.norm(flange))
        fraction = distance / reach

        if fraction > 1.0:
            self.log.error(
                f"{label} pose needs the flange at {distance:.3f} m, beyond the "
                f"arm's {reach:.3f} m reach -- expect MoveIt error -31. "
                "Spawn the die closer to the base, or lower grasp_tilt_deg."
            )
            return False
        if fraction > float(self.node.p("reach_warn_fraction")):
            self.log.warn(
                f"{label} pose puts the flange at {distance:.3f} m, "
                f"{100.0 * fraction:.0f}% of the arm's {reach:.3f} m reach; "
                "IK may fail or take the long way round"
            )
        return True

    # -- choosing a grasp the arm can actually hold ------------------------- #

    def reachable(self, position, quaternion) -> bool:
        """Ask easy_motion's IK whether this pose has a solution at all."""
        return self._solve_ik(position, quaternion, seed=None) is not None

    def choose_grasp(self, move: Regrasp, die_center, die_yaw: float, close_axis: str):
        """Pick a lean and a wrist roll that the arm can reach, not just the ideal one.

        45 degrees is the *geometric* optimum -- it centres the flip's
        90-degree excursion on vertical, so neither end is horizontal. It is not
        automatically an *achievable* orientation. On this cell it was not: IK
        ground for three seconds a time and gave up with -31, three attempts in
        a row, before the arm had moved at all.

        Reach was not the problem (the flange check passed), and neither was
        geometry. A six-axis arm simply cannot hold every orientation at every
        point, and a UR5e reaching across the board has its wrist close to a
        singularity where whole families of orientations drop out.

        So this searches instead of assuming, over two freedoms that cost
        nothing to vary:

        * **the lean**, from the ideal outwards -- 45, then 38, 52, 32, 58.
          Every one of them keeps both ends of the flip clear of horizontal,
          which is the property that matters; the exact angle does not.
        * **the wrist roll**, both of the two 180-degree-apart grasps that grip
          the same faces from the same side. These are genuinely
          interchangeable to the die and can be worlds apart for the wrist.

        The first combination whose four critical poses all solve wins, and the
        preferred one is tried first, so the normal case costs four quick IK
        calls and no extra motion. Ordering matters: the wrist roll nearer the
        arm's current pose comes first, so a working grasp is also a cheap one.
        """
        nominal = abs(float(self.node.p("grasp_tilt_deg")))
        if not bool(self.node.p("verify_ik_before_moving")):
            tilt = tilt_for_turn(move.quarter_turns, math.radians(nominal))
            quaternion = grasp_orientation(move.axis, die_yaw, close_axis, tilt)
            return tilt, nearest_equivalent_grasp(
                quaternion, self.node.lookup_tool_orientation()
            )

        spread = [float(v) for v in self.node.p("grasp_tilt_search_deg")]
        reference = self.node.lookup_tool_orientation()
        attempted = []

        for offset in spread:
            angle = nominal + offset
            if not 20.0 <= angle <= 70.0:
                # Outside this band one end of the flip is close enough to
                # horizontal to bring the gripper down onto the board again.
                continue
            tilt = tilt_for_turn(move.quarter_turns, math.radians(angle))
            base = grasp_orientation(move.axis, die_yaw, close_axis, tilt)
            preferred = nearest_equivalent_grasp(base, reference)
            rolls = [preferred]
            other = flip_about_approach(preferred)
            if not np.allclose(other, preferred):
                rolls.append(other)

            for roll, quaternion in enumerate(rolls):
                unreachable = self._first_unreachable(
                    move, die_center, die_yaw, quaternion
                )
                if unreachable is None:
                    if attempted:
                        self.log.info(
                            f"lean {angle:.0f} deg (roll {roll}) is reachable; "
                            f"tried {', '.join(attempted)} first"
                        )
                    return tilt, quaternion
                attempted.append(f"{angle:.0f}deg/roll{roll}({unreachable})")

        # Nothing solved. Say so clearly, then go anyway with the preferred
        # grasp and let MoveIt deliver the verdict.
        #
        # Refusing to move would be the tidier-looking choice and the wrong one:
        # this check is an *advisory* built on a separate IK service, and if that
        # service is missing or answering badly, every candidate looks
        # unreachable whether or not it is. A pre-flight check that can ground
        # the robot on its own bad day is worse than no pre-flight check.
        self.log.error(
            "no lean passed the IK check: "
            f"{', '.join(attempted) if attempted else 'no candidates in range'}"
        )
        self.log.error(
            "Either the die is somewhere the arm cannot work -- try spawning it "
            "nearer the base, x around -0.1, y around 0.60 -- or the IK service "
            "is not answering. Attempting the nominal grasp anyway; if MoveIt "
            "also reports -31 then it is the former."
        )
        tilt = tilt_for_turn(move.quarter_turns, math.radians(nominal))
        quaternion = grasp_orientation(move.axis, die_yaw, close_axis, tilt)
        return tilt, nearest_equivalent_grasp(quaternion, reference)

    def _first_unreachable(
        self, move: Regrasp, die_center, die_yaw: float, quaternion
    ) -> Optional[str]:
        """Name the first pose of the cycle IK cannot solve, or ``None``."""
        for label, position, orientation in self.cycle_poses(
            move, die_center, die_yaw, quaternion
        ):
            if not self.warn_if_near_reach_limit(label, position, orientation):
                return label
            if not self.reachable(position, orientation):
                return label
        return None

    def cycle_poses(self, move: Regrasp, die_center, die_yaw: float, quaternion):
        """The four poses a re-grasp has to hit, in order.

        Only these are checked. The intermediate lift and the retreat are
        straight vertical moves between two poses already known to be good, so
        if the ends solve the middle does too.
        """
        approach = float(self.node.p("approach_height_m"))
        lift = float(self.node.p("lift_height_m"))
        clearance = float(self.node.p("place_clearance_m"))

        grasp_point = np.asarray(die_center, dtype=float)
        above = grasp_point + np.array([0.0, 0.0, approach])
        lifted = grasp_point + np.array([0.0, 0.0, lift])
        turned_position, turned_quaternion = rotate_about_axis(
            lifted, quaternion, move.axis, move.quarter_turns, lifted, die_yaw
        )
        place = np.array([die_center[0], die_center[1], die_center[2] + clearance])

        return [
            ("pre-grasp", above, quaternion),
            ("grasp", grasp_point, quaternion),
            ("turn", turned_position, turned_quaternion),
            ("place", place, turned_quaternion),
        ]

    # -- one re-grasp ------------------------------------------------------ #

    def execute_regrasp(self, move: Regrasp) -> bool:
        """Pick the die, turn it 90 degrees about ``move.axis``, put it down."""
        pose = self.node.lookup_die_pose()
        if pose is None:
            return False
        die_center, die_yaw = pose

        approach = float(self.node.p("approach_height_m"))
        lift = float(self.node.p("lift_height_m"))
        clearance = float(self.node.p("place_clearance_m"))
        fast = float(self.node.p("velocity_scaling"))
        slow = float(self.node.p("approach_velocity_scaling"))
        close_axis = self.node.close_axis()

        # Lean the gripper away from vertical, in the direction that leaves it
        # leaning the *other* way by the same amount once the turn is done.
        #
        # This is the fix for the run that kept aborting with MoveIt error 99999
        # after a clean pick.  A straight-down grasp has to finish a 90-degree
        # flip pointing horizontally, and a horizontal gripper cannot be lowered
        # to the board -- its own body is in the way.  MoveIt said as much: the
        # Cartesian planner got 75% of the way down and stopped, which is
        # exactly where the gripper reaches the board.  Splitting the excursion
        # either side of vertical means neither end is horizontal and the die
        # can be set down instead of dropped.
        #
        # *Which* lean, though, is not ours to decide alone: 45 degrees is the
        # geometric optimum and the arm may simply not be able to hold it. So
        # ask, rather than assume -- see choose_grasp.
        tilt, quaternion = self.choose_grasp(move, die_center, die_yaw, close_axis)
        if quaternion is None:
            return False

        grasp_point = die_center.copy()
        # Stand off **straight up**, not back along the tool axis.
        #
        # Backing off along a leaning tool axis is the textbook approach -- it
        # slides the fingers on along their own length -- but it also swings the
        # flange outward, and on a UR5e reaching across the board there is no
        # room for that. At this die position it puts the flange at 0.847 m of
        # the arm's 0.850 m reach, and IK simply fails (MoveIt error -31, "no IK
        # solution", on the free-space move before the gripper has gone
        # anywhere). Straight up costs 0.789 m and plans immediately.
        #
        # Nothing is lost by going up instead: the open fingers straddle the die
        # along the grasp axis, so a vertical descent never touches it however
        # far the gripper leans.
        above = grasp_point + np.array([0.0, 0.0, approach])

        self.log.info(
            f"grasping at ({grasp_point[0]:.3f}, {grasp_point[1]:.3f}, "
            f"{grasp_point[2]:.3f}), yaw {math.degrees(die_yaw):.1f} deg, "
            f"lean {math.degrees(tilt):+.0f} deg, {move}"
        )

        # Lift before turning.  Turning at table height is how you drive a
        # corner of the die into the board -- the failure the hands-on slides
        # single out with a red cross.
        lifted = grasp_point + np.array([0.0, 0.0, lift])
        turned_position, turned_quaternion = rotate_about_axis(
            lifted, quaternion, move.axis, move.quarter_turns, lifted, die_yaw
        )
        # Put it back down where it came from, a hair above the board so the die
        # drops the last fraction of a millimetre instead of being pressed.
        place = np.array([die_center[0], die_center[1], die_center[2] + clearance])
        retreat = place + np.array([0.0, 0.0, approach])

        # The cycle splits at the two points where the gripper acts; each piece
        # is a path the robot can run without stopping.
        approach_path = [(above, quaternion), (grasp_point, quaternion)]
        carry_path = [
            (grasp_point, quaternion),
            (lifted, quaternion),
            (turned_position, turned_quaternion),
            (place, turned_quaternion),
        ]
        retreat_path = [(place, turned_quaternion), (retreat, turned_quaternion)]

        # Getting *to* the start of the cycle is a free-space move; plan it with
        # MoveIt either way, so the arm avoids the cell on the way in.
        if not self._move(above, quaternion, cartesian=False, velocity=fast):
            return False

        trajectory_mode = str(self.node.p("motion_mode")).lower() == "trajectory"

        def run(path, fallback_speeds) -> bool:
            if trajectory_mode and self.run_cartesian_path(path):
                return True
            for (position, orientation), (cartesian, velocity) in zip(
                path[1:], fallback_speeds
            ):
                if not self._move(position, orientation, cartesian, velocity):
                    return False
            return True

        if not run(approach_path, [(True, slow)]):
            return False

        self.close_gripper()
        self.motion.attach_object(
            str(self.node.p("attached_object_id")), str(self.node.p("tool_frame"))
        )

        if not run(carry_path, [(True, slow), (False, fast), (True, slow)]):
            self._abort_grasp()
            return False

        self.open_gripper()
        self.motion.detach_object(str(self.node.p("attached_object_id")))

        run(retreat_path, [(True, slow)])

        time.sleep(float(self.node.p("settle_seconds")))
        return True

    def _abort_grasp(self) -> None:
        self.log.warn("aborting grasp: releasing the die")
        self.open_gripper()
        self.motion.detach_object(str(self.node.p("attached_object_id")))

    # -- the loop ---------------------------------------------------------- #

    def run(self) -> RunReport:
        target = int(self.node.p("target_face"))
        if not 1 <= target <= 6:
            raise ValueError(f"target_face must be in 1..6, got {target}")

        strategy = str(self.node.p("strategy"))
        report = RunReport(target_face=target)
        started = time.monotonic()

        if not self.node.wait_for_tf():
            report.seconds = time.monotonic() - started
            return report

        if not self.go_home():
            report.seconds = time.monotonic() - started
            return report

        policy: Optional[DeductivePolicy] = None
        max_regrasps = int(self.node.p("max_regrasps"))

        for index in range(max_regrasps + 1):
            face, _, ok = self.vision.identify()
            if not ok or not 1 <= face <= 6:
                self.log.error("die identification failed")
                report.attempts.append(Attempt(index, face, None, strategy, 0.0, False))
                break

            self.log.info(f"face {face} is up (target {target}, bottom {opposite(face)})")
            if face == target:
                report.success = True
                report.final_face = face
                break

            move, used = self._next_move(face, target, strategy, policy)
            if move is None:
                self.log.error("no move available; giving up")
                report.attempts.append(Attempt(index, face, None, used, 0.0, False))
                break
            if used == "deduce" and policy is None:
                policy = self._policy
            self.log.info(f"plan: {describe_plan([move])} [{used}]")

            step_started = time.monotonic()
            done = self.execute_regrasp(move)
            attempt = Attempt(index, face, move, used, time.monotonic() - step_started, done)
            report.attempts.append(attempt)
            if not done:
                break
        else:
            self.log.error(f"gave up after {max_regrasps} re-grasps")

        if not report.success:
            face, _, ok = self.vision.identify()
            report.final_face = face if ok else 0
            report.success = report.final_face == target

        report.seconds = time.monotonic() - started
        return report

    def _next_move(
        self, face: int, target: int, strategy: str, policy: Optional[DeductivePolicy]
    ) -> Tuple[Optional[Regrasp], str]:
        """Pick the next re-grasp, preferring the exact planner when possible.

        ``strategy`` is one of:

        ``exact``
            Use the simulator's ``face{1..6}_tf`` frames, which give the die's
            full orientation, and drive straight at the target.
        ``deduce`` (``blind``)
            Pretend only the top face is visible -- which is all a real overhead
            camera gives you -- and use :class:`DeductivePolicy`.
        ``auto``
            ``exact`` when the face frames are published, ``deduce`` otherwise.
        """
        if strategy in ("auto", "exact"):
            normals = self.node.face_normals_in_base()
            pose = self.node.lookup_die_pose()
            if normals is not None and pose is not None:
                _, die_yaw = pose
                plan = plan_from_normals(normals_in_grasp_frame(normals, die_yaw), target)
                if plan:
                    return plan[0], "exact"
            if strategy == "exact":
                self.log.error("face frames unavailable and strategy is 'exact'")
                return None, "exact"
            self.log.info("face frames unavailable; deducing from the top face alone")

        if policy is None:
            policy = DeductivePolicy(target)
            self._policy = policy
        move = policy.observe(face)
        self.log.info(f"reasoning: {policy.reasoning}")
        return move, "deduce"


def main(args=None) -> None:
    rclpy.init(args=args)

    node = DiceTaskNode()

    # DiceTaskNode owns the TF listener, and a TF listener only receives /tf
    # while its node is being executed.  MotionClient and VisionClient spin
    # themselves when they make a call, but nothing was spinning this node, so
    # its buffer stayed empty and every lookup failed with "base_link does not
    # exist".  Give it its own executor on a background thread; the other two
    # nodes keep spinning themselves, so there is no contention.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    motion = MotionClient(gripper_action_name="/gripper_action_controller/gripper_cmd")
    vision = VisionClient(str(node.p("identification_service")))

    try:
        challenge = DiceChallenge(node, motion, vision)
        report = challenge.run()
        node.get_logger().info(report.summary())
        for attempt in report.attempts:
            node.get_logger().info(
                f"  step {attempt.index}: saw {attempt.observed_face}, "
                f"{attempt.move or 'no move'} [{attempt.strategy}] "
                f"{attempt.seconds:.1f} s "
                f"{'ok' if attempt.succeeded else 'FAILED'}"
            )
    except Exception as exc:  # noqa: BLE001 - report cleanly rather than trace
        node.get_logger().error(f"dice task aborted: {exc}")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        for n in (vision, motion, node):
            try:
                n.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
