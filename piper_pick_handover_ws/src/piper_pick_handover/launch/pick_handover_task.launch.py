import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("piper_pick_handover")

    moveit_executor_yaml = os.path.join(
        share, "config", "moveit_executor.yaml"
    )
    table_scene_yaml = os.path.join(
        share, "config", "table_scene.yaml"
    )
    observe_hand_yaml = os.path.join(
        share, "config", "observe_hand.yaml"
    )
    handover_yaml = os.path.join(
        share, "config", "handover.yaml"
    )

    moveit_executor = Node(
        package="piper_pick_handover",
        executable="moveit_executor_node",
        name="moveit_executor_node",
        output="screen",
        parameters=[moveit_executor_yaml],
    )

    table_scene = Node(
        package="piper_pick_handover",
        executable="table_scene_node",
        name="table_scene_node",
        output="screen",
        parameters=[table_scene_yaml],
    )

    # Pure MoveIt WORK_HOME controller.
    # No old startup JointState/control_enable path is launched here.
    work_home_controller = Node(
        package="piper_pick_handover",
        executable="work_home_controller_node",
        name="work_home_controller_node",
        output="screen",
    )

    observe_hand_planner = Node(
        package="piper_pick_handover",
        executable="observe_hand_planner_node",
        name="observe_hand_planner_node",
        output="screen",
        parameters=[observe_hand_yaml],
    )

    handover_planner = Node(
        package="piper_pick_handover",
        executable="handover_planner_node",
        name="handover_planner_node",
        output="screen",
        parameters=[handover_yaml],
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
                    "TASK LAYER ONLY: MoveIt executor + table scene + WORK_HOME "
                    "+ observe-hand planner + handover planner + manager."
                )
            ),
            LogInfo(
                msg=(
                    "Requires the PIPER/MoveIt terminal and vision terminal to be running."
                )
            ),
            moveit_executor,
            table_scene,
            work_home_controller,
            observe_hand_planner,
            handover_planner,
            manager,
        ]
    )
