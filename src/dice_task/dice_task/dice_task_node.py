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

``blind``
    Uses only the top face, which is all a single overhead camera can see, and
    closes the loop: turn, look again, turn again.  Two re-grasps on average,
    four in the worst case.  This is the strategy that works on the real cell.

``auto`` (the default) uses ``exact`` when the face frames are available and
falls back to ``blind`` when they are not, so the same node runs in both places.
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
    BlindSearchPolicy,
    FACE_NORMALS,
    Regrasp,
    describe_plan,
    opposite,
    plan_from_normals,
)
from dice_task.grasping import (
    AXIS_VECTORS,
    grasp_orientation,
    normals_in_grasp_frame,
    rotate_about_axis,
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
            return 0, None, False
        return int(result.face_number), result.pose, bool(result.success)


class DiceTaskNode(Node):
    """Owns the parameters, the TF buffer and the state machine."""

    def __init__(self) -> None:
        super().__init__("dice_task")

        self.declare_parameter("target_face", 1)
        self.declare_parameter("strategy", "auto")  # auto | exact | blind
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
        self.declare_parameter("approach_height_m", 0.10)
        self.declare_parameter("lift_height_m", 0.12)
        self.declare_parameter("place_clearance_m", 0.004)
        self.declare_parameter("velocity_scaling", 0.3)
        self.declare_parameter("approach_velocity_scaling", 0.15)
        self.declare_parameter("max_regrasps", 8)
        self.declare_parameter("settle_seconds", 0.7)
        self.declare_parameter("attached_object_id", "dice")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

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
        the face frames are missing and silently drops to the blind policy --
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

    # -- one re-grasp ------------------------------------------------------ #

    def execute_regrasp(self, move: Regrasp) -> bool:
        """Pick the die, turn it 90 degrees about ``move.axis``, put it down."""
        pose = self.node.lookup_die_pose()
        if pose is None:
            return False
        die_center, die_yaw = pose

        dice_size = float(self.node.p("dice_size_m"))
        approach = float(self.node.p("approach_height_m"))
        lift = float(self.node.p("lift_height_m"))
        clearance = float(self.node.p("place_clearance_m"))
        fast = float(self.node.p("velocity_scaling"))
        slow = float(self.node.p("approach_velocity_scaling"))
        close_axis = self.node.close_axis()

        # The fingers must close along the axis the die will turn about, and
        # square onto the die's lateral faces -- hence the die's own yaw.
        quaternion = grasp_orientation(move.axis, die_yaw, close_axis)

        grasp_point = die_center.copy()
        above = grasp_point + np.array([0.0, 0.0, approach])

        self.log.info(
            f"grasping at ({grasp_point[0]:.3f}, {grasp_point[1]:.3f}, "
            f"{grasp_point[2]:.3f}), yaw {math.degrees(die_yaw):.1f} deg, {move}"
        )

        if not self._move(above, quaternion, cartesian=False, velocity=fast):
            return False
        if not self._move(grasp_point, quaternion, cartesian=True, velocity=slow):
            return False

        self.close_gripper()
        self.motion.attach_object(
            str(self.node.p("attached_object_id")), str(self.node.p("tool_frame"))
        )

        # Lift before turning.  Turning at table height is how you drive a
        # corner of the die into the board -- the failure the hands-on slides
        # single out with a red cross.
        lifted = grasp_point + np.array([0.0, 0.0, lift])
        if not self._move(lifted, quaternion, cartesian=True, velocity=slow):
            self._abort_grasp()
            return False

        pivot = lifted  # rotate about the die itself, not about the flange
        turned_position, turned_quaternion = rotate_about_axis(
            lifted, quaternion, move.axis, move.quarter_turns, pivot, die_yaw
        )
        if not self._move(turned_position, turned_quaternion, cartesian=False, velocity=fast):
            self._abort_grasp()
            return False

        # Put it back down where it came from, a hair above the board so the
        # die drops the last fraction of a millimetre instead of being pressed.
        place = np.array(
            [die_center[0], die_center[1], die_center[2] + clearance]
        )
        if not self._move(place, turned_quaternion, cartesian=True, velocity=slow):
            self._abort_grasp()
            return False

        self.open_gripper()
        self.motion.detach_object(str(self.node.p("attached_object_id")))

        retreat = place + np.array([0.0, 0.0, approach])
        self._move(retreat, turned_quaternion, cartesian=True, velocity=slow)

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

        policy: Optional[BlindSearchPolicy] = None
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
            if used == "blind" and policy is None:
                policy = self._blind_policy
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
        self, face: int, target: int, strategy: str, policy: Optional[BlindSearchPolicy]
    ) -> Tuple[Optional[Regrasp], str]:
        """Pick the next re-grasp, preferring the exact planner when possible."""
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
            self.log.info("face frames unavailable; falling back to the blind policy")

        if policy is None:
            policy = BlindSearchPolicy(target)
            self._blind_policy = policy
        return policy.observe(face), "blind"


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
