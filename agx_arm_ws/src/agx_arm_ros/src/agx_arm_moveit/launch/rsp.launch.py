import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from _moveit_config_builder import build_moveit_config, declare_common_args


def _launch(context):
    follow = LaunchConfiguration("follow").perform(context)
    feedback_topic = LaunchConfiguration("feedback_topic").perform(context)
    control_topic = LaunchConfiguration("control_topic").perform(context)
    joint_states_topic = str(feedback_topic) if follow == "true" else str(control_topic)

    moveit_config = build_moveit_config(context)

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            respawn=True,
            output="screen",
            parameters=[moveit_config.robot_description],
            remappings=[("joint_states", joint_states_topic)],
        )
    ]


def generate_launch_description():
    return LaunchDescription(declare_common_args() + [OpaqueFunction(function=_launch)])
