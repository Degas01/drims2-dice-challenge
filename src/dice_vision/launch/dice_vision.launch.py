"""Start the vision node against the DRIMS OAK camera (or a rosbag of it)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "image_topic",
            default_value="/oak/rgb/image_raw/compressed",
            description="Camera topic. The DRIMS bags publish compressed images here.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/oak/rgb/camera_info"
        ),
        DeclareLaunchArgument(
            "compressed",
            default_value="true",
            description="Set false when subscribing to a raw sensor_msgs/Image topic.",
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="false",
            description=(
                "Broadcast the die frame. Leave false when the dice simulator is "
                "running, or two publishers will fight over the same frame."
            ),
        ),
        DeclareLaunchArgument("publish_mask", default_value="false"),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("dice_vision"), "config", "dice_vision.yaml"]
            ),
        ),
    ]

    node = Node(
        package="dice_vision",
        executable="dice_vision_node",
        name="dice_vision",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "compressed": LaunchConfiguration("compressed"),
                "publish_tf": LaunchConfiguration("publish_tf"),
                "publish_mask": LaunchConfiguration("publish_mask"),
            },
        ],
        remappings=[
            ("~/image", LaunchConfiguration("image_topic")),
            ("~/camera_info", LaunchConfiguration("camera_info_topic")),
        ],
    )

    return LaunchDescription(arguments + [node])
