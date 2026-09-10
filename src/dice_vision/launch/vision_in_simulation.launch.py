"""Run the whole perception chain against the DRIMS *simulation*.

The simulation has no camera, so this starts a synthetic one that watches the
simulated die and renders the overhead view a real camera would have seen, then
points the vision node at it. Everything downstream is unchanged, so this
exercises the real detector, the real camera model and the real service.

Expects the cell and the die to be up already:

    ros2 launch drims_description ur5e_1_start.launch.py fake:=true
    ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=5

Then:

    ros2 launch dice_vision vision_in_simulation.launch.py die_colour:=blue
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "die_colour",
            default_value="yellow",
            description=(
                "yellow, red, blue, green_dark, orange, black, white or purple. "
                "The detector is colour-agnostic; this is how you prove it."
            ),
        ),
        DeclareLaunchArgument("dice_size_m", default_value="0.03"),
        DeclareLaunchArgument("camera_height_m", default_value="1.0"),
        DeclareLaunchArgument(
            "board_origin_in_base",
            default_value="[-0.05, 0.675, -0.02]",
            description="Board centre in base_link. Must match the active cell.",
        ),
        DeclareLaunchArgument(
            "dist_coeffs",
            default_value="[0.0, 0.0, 0.0, 0.0, 0.0]",
            description="Lens distortion. Try [-0.28, 0.09, 0.0, 0.0, 0.0].",
        ),
        DeclareLaunchArgument("shadow_strength", default_value="0.45"),
        DeclareLaunchArgument("light_gradient", default_value="0.25"),
        DeclareLaunchArgument("rate_hz", default_value="10.0"),
    ]

    shared = {
        "board_origin_in_base": LaunchConfiguration("board_origin_in_base"),
        "camera_height_m": LaunchConfiguration("camera_height_m"),
        "dice_size_m": LaunchConfiguration("dice_size_m"),
    }

    fake_camera = Node(
        package="dice_vision",
        executable="fake_camera_node",
        name="fake_camera",
        output="screen",
        parameters=[
            shared,
            {
                "die_colour": LaunchConfiguration("die_colour"),
                "dist_coeffs": LaunchConfiguration("dist_coeffs"),
                "shadow_strength": LaunchConfiguration("shadow_strength"),
                "light_gradient": LaunchConfiguration("light_gradient"),
                "rate_hz": LaunchConfiguration("rate_hz"),
            },
        ],
    )

    vision = Node(
        package="dice_vision",
        executable="dice_vision_node",
        name="dice_vision",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("dice_vision"), "config", "dice_vision.yaml"]
            ),
            shared,
            {
                "compressed": True,
                "publish_tf": False,
            },
        ],
        remappings=[
            ("~/image", "/fake_camera/image_raw/compressed"),
            ("~/camera_info", "/fake_camera/camera_info"),
        ],
    )

    return LaunchDescription(arguments + [fake_camera, vision])
