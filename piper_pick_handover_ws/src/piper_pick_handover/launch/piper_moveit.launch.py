import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _set_move_group_start_tolerance(context):
    """
    Set MoveIt's trajectory start-state tolerance after /move_group starts.

    The AgileX included launch creates /move_group internally, so this project
    launch cannot directly inject the parameter into that Node action.  Instead,
    retry `ros2 param set` until /move_group is ready.

    This runs automatically on every launch.
    """
    tolerance = LaunchConfiguration(
        "allowed_start_tolerance"
    ).perform(context)

    command = (
        "for i in $(seq 1 20); do "
        "if ros2 param set /move_group "
        "trajectory_execution.allowed_start_tolerance "
        f"{tolerance}; then "
        "echo '[piper_moveit] move_group allowed_start_tolerance="
        f"{tolerance}'; "
        "exit 0; "
        "fi; "
        "sleep 0.5; "
        "done; "
        "echo '[piper_moveit] ERROR: failed to set "
        "trajectory_execution.allowed_start_tolerance' >&2; "
        "exit 1"
    )

    return [
        ExecuteProcess(
            cmd=[
                "bash",
                "-lc",
                command,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    can_port = LaunchConfiguration("can_port")
    arm_type = LaunchConfiguration("arm_type")
    effector_type = LaunchConfiguration("effector_type")
    fw_version = LaunchConfiguration("fw_version")
    speed_percent = LaunchConfiguration("speed_percent")
    auto_enable = LaunchConfiguration("auto_enable")
    follow = LaunchConfiguration("follow")
    auto_control_gate = LaunchConfiguration("auto_control_gate")
    allowed_start_tolerance = LaunchConfiguration("allowed_start_tolerance")

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
            DeclareLaunchArgument(
                "allowed_start_tolerance",
                default_value="0.03",
                description=(
                    "MoveIt trajectory_execution.allowed_start_tolerance [rad]. "
                    "0.03 rad is used for the real PIPER to tolerate small "
                    "planning/execution start-state differences."
                ),
            ),
            LogInfo(
                msg=(
                    "PIPER MOTION LAYER ONLY: real PIPER + MoveIt + RViz + "
                    "automatic control gate. No vision/project task nodes."
                )
            ),
            LogInfo(
                msg=[
                    "MoveIt allowed_start_tolerance will be forced to ",
                    allowed_start_tolerance,
                    " rad after /move_group becomes available.",
                ]
            ),

            agx_moveit,

            # /move_group is created inside the included AgileX launch.
            # Start retrying after a short delay; the helper itself retries
            # for up to ~10 seconds if /move_group is not ready yet.
            TimerAction(
                period=3.0,
                actions=[
                    OpaqueFunction(
                        function=_set_move_group_start_tolerance
                    )
                ],
            ),
        ]
    )
