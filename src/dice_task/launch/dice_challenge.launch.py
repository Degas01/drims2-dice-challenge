"""Run the dice challenge: bring a requested face of the die to the top.

Expects the robot cell and the die to be up already, e.g.

    ros2 launch drims_description ur5e_1_start.launch.py fake:=true
    ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "target_face", default_value="1", description="Face to expose, 1-6."
        ),
        DeclareLaunchArgument(
            "strategy",
            default_value="auto",
            description=(
                "auto: use the face TF frames when present, else close the loop "
                "on the camera alone. exact: require the face frames. "
                "blind: camera only, as on the real cell."
            ),
        ),
        DeclareLaunchArgument("dice_size_m", default_value="0.03"),
        DeclareLaunchArgument(
            "motion_mode",
            default_value="moveit",
            description=(
                "moveit: one move_to_pose per waypoint, each planned to a full "
                "stop, every motion collision-checked. trajectory: generate the "
                "Cartesian path here, blend the corners and send one timed "
                "JointTrajectory -- faster, but it bypasses MoveIt's "
                "planning-scene checks."
            ),
        ),
        DeclareLaunchArgument(
            "blend_radius_m",
            default_value="0.03",
            description="Corner rounding for motion_mode:=trajectory. 0 disables it.",
        ),
        DeclareLaunchArgument("max_regrasps", default_value="8"),
        DeclareLaunchArgument("velocity_scaling", default_value="0.3"),
        DeclareLaunchArgument(
            "identification_service",
            default_value="dice_identification",
            description=(
                "The simulator provides this; point it at the dice_vision node "
                "instead to run the challenge off the real camera."
            ),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("dice_task"), "config", "dice_task.yaml"]
            ),
        ),
    ]

    node = Node(
        package="dice_task",
        executable="dice_task_node",
        name="dice_task",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "target_face": LaunchConfiguration("target_face"),
                "strategy": LaunchConfiguration("strategy"),
                "dice_size_m": LaunchConfiguration("dice_size_m"),
                "motion_mode": LaunchConfiguration("motion_mode"),
                "blend_radius_m": LaunchConfiguration("blend_radius_m"),
                "max_regrasps": LaunchConfiguration("max_regrasps"),
                "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                "identification_service": LaunchConfiguration("identification_service"),
            },
        ],
    )

    return LaunchDescription(arguments + [node])
