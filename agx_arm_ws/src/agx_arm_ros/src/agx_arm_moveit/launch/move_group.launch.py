import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg

from _moveit_config_builder import build_moveit_config, declare_common_args


def _launch(context):
    moveit_config = build_moveit_config(context)
    follow = LaunchConfiguration("follow").perform(context) == "true"
    feedback_topic = LaunchConfiguration("feedback_topic").perform(context)
    control_topic = LaunchConfiguration("control_topic").perform(context)
    joint_states_topic = str(feedback_topic) if follow else str(control_topic)

    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(
            LaunchConfiguration("capabilities"), value_type=str
        ),
        "disable_capabilities": ParameterValue(
            LaunchConfiguration("disable_capabilities"), value_type=str
        ),
        "publish_planning_scene": LaunchConfiguration(
            "publish_monitored_planning_scene"
        ),
        "publish_geometry_updates": LaunchConfiguration(
            "publish_monitored_planning_scene"
        ),
        "publish_state_updates": LaunchConfiguration(
            "publish_monitored_planning_scene"
        ),
        "publish_transforms_updates": LaunchConfiguration(
            "publish_monitored_planning_scene"
        ),
        "monitor_dynamics": False,
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
    ]

    remappings = [("joint_states", joint_states_topic)]

    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_params,
            remappings=remappings,
            additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
            condition=UnlessCondition(LaunchConfiguration("debug")),
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_params,
            remappings=remappings,
            prefix=["gdb -x {} --ex run --args".format(
                moveit_config.package_path / "launch" / "gdb_settings.gdb"
            )],
            additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
            condition=IfCondition(LaunchConfiguration("debug")),
        ),
    ]


def generate_launch_description():
    from launch.actions import DeclareLaunchArgument

    return LaunchDescription(
        declare_common_args()
        + [
            DeclareBooleanLaunchArg("debug", default_value=False),
            DeclareBooleanLaunchArg(
                "allow_trajectory_execution", default_value=True
            ),
            DeclareBooleanLaunchArg(
                "publish_monitored_planning_scene", default_value=True
            ),
            DeclareBooleanLaunchArg("monitor_dynamics", default_value=False),
            DeclareLaunchArgument("capabilities", default_value=""),
            DeclareLaunchArgument("disable_capabilities", default_value=""),
            OpaqueFunction(function=_launch),
        ]
    )
