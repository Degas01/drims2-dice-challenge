"""A synthetic overhead camera, so the vision pipeline can run in simulation.

The DRIMS simulation has no camera.  ``ur5e_N_start.launch.py fake:=true`` brings
up the arm and MoveIt, and ``drims_dice_simulator`` publishes the die's *ground
truth* pose and face directly on ``/dice_identification`` -- there is no image
anywhere.  So the perception half of this project could only ever be exercised
offline against the test-suite, or against the recorded rosbags.

This node closes that gap.  It watches the simulated die -- its face on
``/dice_face`` and its pose on TF -- renders the overhead view that a camera
above the board *would* have seen, and publishes it as a normal
``sensor_msgs/CompressedImage`` plus ``CameraInfo``.  Point ``dice_vision`` at
those topics and the whole chain runs closed-loop in simulation:

    dice simulator -> fake camera -> detector -> pose -> state machine -> robot
                 ^                                                          |
                 +----------------------------------------------------------+

That makes two things testable that otherwise were not:

* **the vision node itself**, live, without hardware or bags;
* **different die colours and sizes**, by changing one parameter -- which is the
  thing the hands-on slides warn about ("yellow (**maybe**)") and the reason the
  detector never looks for a colour in the first place.

It is a *simulated sensor*, not a claim about the real one: the renderer models
projection, perspective, lighting gradient, cast shadow, sensor noise and JPEG
compression, but it cannot stand in for the real board's lighting or the real
lens.  Running the detector on the provided bags is still the honest final check.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Int16
from tf2_ros import Buffer, TransformListener

from dice_vision.scene_simulator import DIE_PALETTE, SceneConfig, render


class FakeCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("fake_camera")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("die_frame", "dice_com_tf")
        self.declare_parameter("frame_id", "oak_rgb_camera_optical_frame")

        # Must match dice_vision's board_origin_in_base, or the two disagree
        # about where the board is and every position comes out shifted.
        self.declare_parameter("board_origin_in_base", [-0.05, 0.675, -0.02])
        self.declare_parameter("board_yaw_in_base", 0.0)
        self.declare_parameter("camera_height_m", 1.0)
        self.declare_parameter("dice_size_m", 0.03)

        # The interesting knobs: change these while everything is running.
        self.declare_parameter("die_colour", "yellow")
        self.declare_parameter("board_bgr", [70, 175, 95])
        self.declare_parameter("light_gradient", 0.25)
        self.declare_parameter("shadow_strength", 0.45)
        self.declare_parameter("noise_sigma", 2.5)
        self.declare_parameter("jpeg_quality", 88)
        self.declare_parameter("clutter", True)
        self.declare_parameter("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("focal_px", 900.0)
        self.declare_parameter("publish_raw", False)

        # Used until the die simulator says otherwise, so the node produces a
        # usable image even on its own.
        self.declare_parameter("default_face", 5)
        self.declare_parameter("default_xy_m", [0.0, 0.0])

        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._face: int = int(self.p("default_face"))
        self._seen_die = False

        self._compressed_pub = self.create_publisher(
            CompressedImage, "~/image_raw/compressed", 1
        )
        self._raw_pub = self.create_publisher(Image, "~/image_raw", 1)
        self._info_pub = self.create_publisher(CameraInfo, "~/camera_info", 1)
        self.create_subscription(Int16, "/dice_face", self._on_face, 10)

        period = 1.0 / max(1e-3, float(self.p("rate_hz")))
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"fake camera publishing a {self.p('die_colour')} die at "
            f"{self.p('rate_hz')} Hz on ~/image_raw/compressed"
        )

    def p(self, name: str):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ #

    def _on_face(self, msg: Int16) -> None:
        if 1 <= int(msg.data) <= 6 and int(msg.data) != self._face:
            self._face = int(msg.data)
            self.get_logger().info(f"die simulator reports face {self._face} up")

    def _die_on_board(self) -> Tuple[float, float, float]:
        """Die position in board coordinates and its yaw, from TF.

        Falls back to the configured default when the die simulator is not
        running, so the node is still useful on its own.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.p("base_frame")), str(self.p("die_frame")), rclpy.time.Time()
            )
        except Exception:  # noqa: BLE001 - absence is an expected state
            if self._seen_die:
                self.get_logger().warn(
                    f"lost {self.p('die_frame')}; falling back to the default position",
                    throttle_duration_sec=5.0,
                )
            default = [float(v) for v in self.p("default_xy_m")]
            return default[0], default[1], 0.0

        self._seen_die = True
        t = transform.transform.translation
        q = transform.transform.rotation

        origin = [float(v) for v in self.p("board_origin_in_base")]
        board_yaw = float(self.p("board_yaw_in_base"))
        # Inverse of board_geometry.board_to_base.
        dx, dy = t.x - origin[0], t.y - origin[1]
        cos, sin = math.cos(-board_yaw), math.sin(-board_yaw)
        x = cos * dx - sin * dy
        y = sin * dx + cos * dy

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy) - board_yaw
        return x, y, yaw

    def _tick(self) -> None:
        colour = str(self.p("die_colour"))
        if colour not in DIE_PALETTE:
            self.get_logger().warn(
                f"unknown die_colour {colour!r}; known: {sorted(DIE_PALETTE)}",
                throttle_duration_sec=10.0,
            )
            colour = "yellow"

        x, y, yaw = self._die_on_board()
        width, height = int(self.p("image_width")), int(self.p("image_height"))

        scene = render(
            SceneConfig(
                face_value=self._face,
                die_colour=colour,
                die_size_m=float(self.p("dice_size_m")),
                die_xy_m=(x, y),
                die_yaw_rad=yaw,
                image_size=(width, height),
                focal_px=float(self.p("focal_px")),
                camera_height_m=float(self.p("camera_height_m")),
                board_bgr=tuple(int(v) for v in self.p("board_bgr")),
                light_gradient=float(self.p("light_gradient")),
                shadow_strength=float(self.p("shadow_strength")),
                noise_sigma=float(self.p("noise_sigma")),
                jpeg_quality=int(self.p("jpeg_quality")),
                clutter=bool(self.p("clutter")),
                dist_coeffs=tuple(float(v) for v in self.p("dist_coeffs")),
                seed=int(self.get_clock().now().nanoseconds % 100000),
            )
        )

        stamp = self.get_clock().now().to_msg()
        frame_id = str(self.p("frame_id"))

        compressed = self._bridge.cv2_to_compressed_imgmsg(scene.image, dst_format="jpg")
        compressed.header.stamp = stamp
        compressed.header.frame_id = frame_id
        self._compressed_pub.publish(compressed)

        if self.p("publish_raw"):
            raw = self._bridge.cv2_to_imgmsg(scene.image, encoding="bgr8")
            raw.header.stamp = stamp
            raw.header.frame_id = frame_id
            self._raw_pub.publish(raw)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = frame_id
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.d = [float(v) for v in self.p("dist_coeffs")]
        focal = float(self.p("focal_px"))
        info.k = [focal, 0.0, width / 2.0, 0.0, focal, height / 2.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            focal, 0.0, width / 2.0, 0.0,
            0.0, focal, height / 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self._info_pub.publish(info)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeCameraNode()
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
