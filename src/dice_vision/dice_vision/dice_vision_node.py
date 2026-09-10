"""ROS 2 node exposing the die detector as a ``DiceIdentification`` service.

This is the node the DRIMS hands-on asks for: subscribe to the camera, find the
die, read its top face, and answer ``/dice_identification`` with the face number
and the die's pose.  It is a thin wrapper -- all the perception lives in
:mod:`dice_vision.detector` and :mod:`dice_vision.board_geometry`, which have no
ROS dependency and are covered by the offline test-suite.

Topics
------
``~/image`` (subscribed, remapped to ``/oak/rgb/image_raw/compressed`` by the
launch file) accepts either ``sensor_msgs/CompressedImage`` or
``sensor_msgs/Image``; the type is chosen with the ``compressed`` parameter.

Five debug topics publish the pipeline one stage at a time, which is how the
hands-on session builds it:

===================  ==========================================================
``~/board_mask``     the segmented green board
``~/object_mask``    everything on the board that is not board-coloured, shown
                     in its own colours rather than as a white blob
``~/bounding_box``   the oriented bounding box and centre
``~/overlay``        the readout: face value, die colour (name and swatch),
                     position on the board and yaw.  Also published as
                     ``~/debug_image``
``~/mosaic``         all four tiled and labelled, for a single rviz panel
===================  ==========================================================

Each is rendered only while something is subscribed, so leaving them all
enabled costs nothing.

Services
--------
``dice_identification`` (``easy_motion_msgs/srv/DiceIdentification``) returns the
face number and a ``PoseStamped`` in ``base_frame``.

TF
--
Optionally broadcasts the die frame so the task node can plan grasps against it
exactly as it does against the simulator's ``dice_tf``.  It is **off by default**
so that running this node alongside ``drims_dice_simulator`` does not produce two
publishers fighting over the same frame.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformBroadcaster

from cv_bridge import CvBridge

from easy_motion_msgs.srv import DiceIdentification

from dice_vision.board_geometry import (
    DEFAULT_BOARD_SIZE_M,
    HomographyMapper,
    board_to_base,
)
from dice_vision.camera_model import CameraModel
from dice_vision.detector import DetectorConfig, DiceDetector
from dice_vision.pose_estimation import estimate_top_face_pose


class DiceVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("dice_vision")

        self.declare_parameter("compressed", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("die_frame", "dice_vision_tf")

        # Board geometry.  Defaults come from the DRIMS vision hands-on: the
        # board is roughly 700 x 500 mm with the origin at its centre.
        self.declare_parameter("board_size_m", list(DEFAULT_BOARD_SIZE_M))
        self.declare_parameter("board_origin_in_base", [0.60, 0.10, -0.01])
        self.declare_parameter("board_yaw_in_base", 0.0)
        self.declare_parameter("camera_height_m", 1.0)
        self.declare_parameter("dice_size_m", 0.027)
        self.declare_parameter("recalibrate_every_frame", False)

        # Detector tunables worth exposing; the rest live in DetectorConfig.
        self.declare_parameter("board_hue_min", 30)
        self.declare_parameter("board_hue_max", 95)
        self.declare_parameter("chroma_tolerance", 0.055)
        self.declare_parameter("min_face_confidence", 0.25)

        # Undistort before detecting. The hands-on slides ask for it, and on a
        # wide-angle lens it is worth several millimetres at the board edge.
        self.declare_parameter("undistort", True)
        # Cross-check the homography's answer against PnP on the die's top face.
        # PnP is not used for position -- its range comes from apparent size,
        # which on a ~25 px die is biased by roughly 9% -- but its reprojection
        # residual is a genuine quality signal, and its yaw is good to ~1.5 deg.
        self.declare_parameter("use_pnp_check", True)
        self.declare_parameter("max_reprojection_error_px", 3.0)

        self._bridge = CvBridge()
        self._detector = DiceDetector(self._detector_config())
        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._mapper: Optional[HomographyMapper] = None
        self._principal_point: Optional[tuple] = None
        self._camera: Optional[CameraModel] = None

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        if self.get_parameter("compressed").value:
            self.create_subscription(
                CompressedImage, "~/image", self._on_compressed, sensor_qos
            )
        else:
            self.create_subscription(Image, "~/image", self._on_image, sensor_qos)
        self.create_subscription(CameraInfo, "~/camera_info", self._on_info, sensor_qos)

        # One topic per stage of the pipeline, in the order the hands-on session
        # builds them.  Each is only rendered when something is subscribed, so
        # leaving them all declared costs nothing when nobody is looking.
        self._stage_pubs = {
            name: self.create_publisher(Image, f"~/{name}", 1)
            for name in ("board_mask", "object_mask", "bounding_box", "overlay", "mosaic")
        }
        # The finished overlay under its conventional name as well, so existing
        # rviz layouts and the launch file keep working.
        self._stage_pubs["debug_image"] = self.create_publisher(Image, "~/debug_image", 1)
        self._broadcaster = TransformBroadcaster(self)

        self._service = self.create_service(
            DiceIdentification, "dice_identification", self._on_request
        )

        self.get_logger().info(
            "dice_vision ready; waiting for images on the remapped ~/image topic"
        )

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def _detector_config(self) -> DetectorConfig:
        return DetectorConfig(
            board_hue_range=(
                int(self.get_parameter("board_hue_min").value),
                int(self.get_parameter("board_hue_max").value),
            ),
            chroma_tolerance=float(self.get_parameter("chroma_tolerance").value),
        )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_compressed(self, msg: CompressedImage) -> None:
        try:
            frame = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - never kill the subscription
            self.get_logger().warn(f"could not decode compressed image: {exc}")
            return
        self._store(frame, msg.header)

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"could not convert image: {exc}")
            return
        self._store(frame, msg.header)

    def _on_info(self, msg: CameraInfo) -> None:
        # The principal point is where the camera looks straight down, which is
        # what the parallax correction and the top-face recovery both need.
        self._principal_point = (float(msg.k[2]), float(msg.k[5]))
        self._detector.cfg.nadir_px = self._principal_point
        if self._camera is None:
            self._camera = CameraModel.from_camera_info(
                msg.k, msg.d, (int(msg.width), int(msg.height))
            )
            self.get_logger().info(
                f"camera model: f=({self._camera.fx:.1f}, {self._camera.fy:.1f}) px, "
                f"principal point {self._principal_point}, "
                f"{'with' if self._camera.has_distortion else 'no'} distortion"
            )

    def _rectify(self, frame: np.ndarray) -> np.ndarray:
        """Undistort if we have intrinsics and were asked to."""
        if (
            self._camera is None
            or not self._camera.has_distortion
            or not self.get_parameter("undistort").value
        ):
            return frame
        return self._camera.undistort_image(frame)

    def _store(self, frame: np.ndarray, header) -> None:
        frame = self._rectify(frame)
        with self._lock:
            self._latest_bgr = frame
            self._latest_stamp = header.stamp

        detection = self._detector.detect(frame)
        self._publish_stages(frame, detection, header)

        if detection is not None and self.get_parameter("publish_tf").value:
            pose = self._pose_from_detection(frame, detection)
            if pose is not None:
                self._broadcast(pose)

    def _board_xy(self, frame: np.ndarray, detection) -> Optional[tuple]:
        """The die's position on the board, in metres, for the overlay."""
        if detection is None:
            return None
        mapper = self._ensure_mapper(frame)
        if mapper is None:
            return None
        x, y, _ = mapper.pixel_to_board(
            detection.center_px, height=float(self.get_parameter("dice_size_m").value)
        )
        return float(x), float(y)

    def _publish_stages(self, frame: np.ndarray, detection, header) -> None:
        """Publish whichever pipeline stages somebody is actually watching.

        The stages are the ones the vision hands-on walks through -- the board
        mask, the die pixels alone, the oriented bounding box, and the finished
        readout -- and they exist for the same reason the session introduces
        them one at a time: when a detection is wrong, the stage where it first
        goes wrong tells you why, and a single final overlay does not.
        """
        wanted = {
            name: pub
            for name, pub in self._stage_pubs.items()
            if pub.get_subscription_count() > 0
        }
        if not wanted:
            return

        # The full overlay is the only stage that needs the board homography,
        # and only to print the position; skip the work if nothing wants it.
        needs_overlay = {"overlay", "debug_image"} & set(wanted)
        board_xy = self._board_xy(frame, detection) if needs_overlay else None

        if wanted.keys() <= {"overlay", "debug_image"}:
            images = {"overlay": self._detector.annotate(frame, detection, board_xy)}
        else:
            images = self._detector.stages(frame, detection, board_xy)
        images["debug_image"] = images["overlay"]

        for name, pub in wanted.items():
            image = images.get(name)
            if image is None:
                continue
            message = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            message.header = header
            pub.publish(message)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    def _ensure_mapper(self, frame: np.ndarray) -> Optional[HomographyMapper]:
        """Calibrate the image-to-board homography from the board's own corners.

        The board is a fixed, high-contrast rectangle of known size, so there is
        no reason to hand-click calibration points: the detector already finds
        its outline.  The homography is cached because the camera is bolted in
        place; set ``recalibrate_every_frame`` if the camera can move.
        """
        if self._mapper is not None and not self.get_parameter(
            "recalibrate_every_frame"
        ).value:
            return self._mapper

        quad = self._detector.board_quad(frame)
        if quad is None:
            return None

        height, width = frame.shape[:2]
        nadir = self._principal_point or (width / 2.0, height / 2.0)
        board_size = [float(v) for v in self.get_parameter("board_size_m").value]

        self._mapper = HomographyMapper.from_board_quad(
            quad,
            (board_size[0], board_size[1]),
            camera_height_m=float(self.get_parameter("camera_height_m").value),
            nadir_px=nadir,
        )
        self.get_logger().info("calibrated image-to-board homography from board corners")
        return self._mapper

    def _pose_from_detection(self, frame: np.ndarray, detection) -> Optional[PoseStamped]:
        mapper = self._ensure_mapper(frame)
        if mapper is None:
            self.get_logger().warn("board not visible; cannot localise the die")
            return None

        dice_size = float(self.get_parameter("dice_size_m").value)

        # The camera sees the *top* face, one die-edge above the board, so the
        # ray has to be intersected at that height rather than at the board.
        x, y, _ = mapper.pixel_to_board(detection.center_px, height=dice_size)

        # Recover the yaw in board coordinates rather than in pixels: the camera
        # can be mounted with either handedness, and reading the yaw off the
        # image would silently mirror it.
        length = 0.5 * detection.size_px
        tip = (
            detection.center_px[0] + length * np.cos(detection.yaw_rad),
            detection.center_px[1] + length * np.sin(detection.yaw_rad),
        )
        tx, ty, _ = mapper.pixel_to_board(tip, height=dice_size)
        yaw = float(np.arctan2(ty - y, tx - x))
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # square: modulo 90 deg

        origin = [float(v) for v in self.get_parameter("board_origin_in_base").value]
        board_yaw = float(self.get_parameter("board_yaw_in_base").value)
        bx, by, bz = board_to_base((x, y, 0.0), origin, board_yaw)

        pose = PoseStamped()
        pose.header.frame_id = str(self.get_parameter("base_frame").value)
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = bx
        pose.pose.position.y = by
        # Report the centre of the die, not the centre of its top face.
        pose.pose.position.z = bz + 0.5 * dice_size

        total_yaw = yaw + board_yaw
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = float(np.sin(0.5 * total_yaw))
        pose.pose.orientation.w = float(np.cos(0.5 * total_yaw))
        return pose

    def _broadcast(self, pose: PoseStamped) -> None:
        transform = TransformStamped()
        transform.header = pose.header
        transform.child_frame_id = str(self.get_parameter("die_frame").value)
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation = pose.pose.orientation
        self._broadcaster.sendTransform(transform)

    # ------------------------------------------------------------------ #
    # Service
    # ------------------------------------------------------------------ #

    def _on_request(self, request, response):
        del request
        with self._lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()

        if frame is None:
            self.get_logger().warn("dice_identification called before any image arrived")
            response.success = False
            response.face_number = 0
            return response

        detection = self._detector.detect(frame)
        if detection is None:
            self.get_logger().warn("no die found on the board")
            response.success = False
            response.face_number = 0
            return response

        minimum = float(self.get_parameter("min_face_confidence").value)
        if not detection.found_face or detection.face_confidence < minimum:
            self.get_logger().warn(
                f"die found but the face is unreadable "
                f"(value {detection.face_value}, confidence "
                f"{detection.face_confidence:.2f} < {minimum:.2f})"
            )
            response.success = False
            response.face_number = 0
            return response

        pose = self._pose_from_detection(frame, detection)
        if pose is None:
            response.success = False
            response.face_number = 0
            return response

        # PnP cross-check: an implausible reprojection residual means the quad
        # is not a projected square, so the detection is geometry we should not
        # trust even though the pips read cleanly.
        if self.get_parameter("use_pnp_check").value and self._camera is not None:
            limit = float(self.get_parameter("max_reprojection_error_px").value)
            square = estimate_top_face_pose(
                detection.quad,
                float(self.get_parameter("dice_size_m").value),
                self._camera.without_distortion()
                if self.get_parameter("undistort").value
                else self._camera,
                max_reprojection_error=limit,
            )
            if square is None:
                self.get_logger().warn(
                    "PnP rejected the detected quad: it is not a projected square "
                    f"within {limit:.1f} px. Reporting failure rather than a pose."
                )
                response.success = False
                response.face_number = 0
                return response
            self.get_logger().debug(
                f"PnP agrees: residual {square.reprojection_error:.2f} px, "
                f"yaw {np.degrees(square.yaw_about_normal()):.1f} deg"
            )

        response.face_number = int(detection.face_value)
        response.pose = pose
        response.success = True
        self.get_logger().info(
            f"{detection.colour_name} die, face {detection.face_value} at "
            f"({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}) "
            f"confidence {detection.face_confidence:.2f}"
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiceVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
