import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    can_port = LaunchConfiguration("can_port")
    arm_type = LaunchConfiguration("arm_type")
    effector_type = LaunchConfiguration("effector_type")
    fw_version = LaunchConfiguration("fw_version")
    speed_percent = LaunchConfiguration("speed_percent")
    auto_enable = LaunchConfiguration("auto_enable")
    follow = LaunchConfiguration("follow")
    auto_control_gate = LaunchConfiguration("auto_control_gate")

    agx_ctrl_share = get_package_share_directory("agx_arm_ctrl")

    agx_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                agx_ctrl_share,
                "launch",
                "start_single_agx_arm_moveit.launch.py",
            )
        ),
        launch_arguments={
            "can_port": can_port,
            "arm_type": arm_type,
            "effector_type": effector_type,
            "fw_version": fw_version,
            "speed_percent": speed_percent,
            "auto_enable": auto_enable,
            "follow": follow,
            "auto_control_gate": auto_control_gate,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("can_port", default_value="can0"),
            DeclareLaunchArgument("arm_type", default_value="piper_x"),
            DeclareLaunchArgument("effector_type", default_value="agx_gripper"),
            DeclareLaunchArgument("fw_version", default_value="v189"),
            DeclareLaunchArgument("speed_percent", default_value="10"),
            DeclareLaunchArgument("auto_enable", default_value="true"),
            DeclareLaunchArgument("follow", default_value="true"),
            DeclareLaunchArgument("auto_control_gate", default_value="true"),
            LogInfo(
                msg=(
                    "PIPER MOTION LAYER ONLY: real PIPER + MoveIt + RViz + "
                    "automatic control gate. No vision/project task nodes."
                )
            ),
            agx_moveit,
        ]
    )
