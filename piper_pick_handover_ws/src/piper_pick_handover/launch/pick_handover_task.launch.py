import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory(
        "piper_pick_handover"
    )

    moveit_executor_yaml = os.path.join(
        share,
        "config",
        "moveit_executor.yaml",
    )

    table_scene_yaml = os.path.join(
        share,
        "config",
        "table_scene.yaml",
    )

    handover_yaml = os.path.join(
        share,
        "config",
        "handover.yaml",
    )

    moveit_executor = Node(
        package="piper_pick_handover",
        executable="moveit_executor_node",
        name="moveit_executor_node",
        output="screen",
        parameters=[
            moveit_executor_yaml,
        ],
    )

    table_scene = Node(
        package="piper_pick_handover",
        executable="table_scene_node",
        name="table_scene_node",
        output="screen",
        parameters=[
            table_scene_yaml,
        ],
    )

    work_home_controller = Node(
        package="piper_pick_handover",
        executable="work_home_controller_node",
        name="work_home_controller_node",
        output="screen",
    )

    handover_planner = Node(
        package="piper_pick_handover",
        executable="handover_planner_node",
        name="handover_planner_node",
        output="screen",
        parameters=[
            handover_yaml,
        ],
    )

    manager = Node(
        package="piper_pick_handover",
        executable="manager_node",
        name="piper_manager_node",
        output="screen",
    )

    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    "TASK LAYER: MoveIt executor + table scene + WORK_HOME "
                    "+ handover planner + manager."
                )
            ),
            LogInfo(
                msg=(
                    "Post-Lift flow: WORK_HOME while holding object -> "
                    "fresh Palm Final -> Handover. "
                    "observe_hand_planner_node is no longer used."
                )
            ),
            LogInfo(
                msg=(
                    "Requires the PIPER/MoveIt terminal and vision terminal "
                    "to already be running."
                )
            ),

            moveit_executor,
            table_scene,
            work_home_controller,
            handover_planner,
            manager,
        ]
    )
