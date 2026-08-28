import json
import os
from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # ============================================================
    # piper_handshake package
    # ============================================================
    handshake_share = (
        get_package_share_directory(
            "piper_handshake"
        )
    )
    calibration_file = os.path.join(
        handshake_share,
        "config",
        "handeye_calibration.json"
    )

    # ============================================================
    # Load Eye-in-Hand calibration
    # ============================================================
    with open(
        calibration_file,
        "r",
        encoding="utf-8"
    ) as f:

        calibration = json.load(f)
    x, y, z = calibration["position"]
    qx, qy, qz, qw = (
        calibration["orientation"]
    )
    return LaunchDescription([

        # ========================================================
        # 1. Eye-in-Hand static TF
        #
        # gripper_base
        #       ↓
        # camera_color_optical_frame
        #
        # 以后不用手动 static_transform_publisher
        # ========================================================
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="handeye_static_tf",
            arguments=[
                "--x", str(x),
                "--y", str(y),
                "--z", str(z),

                "--qx", str(qx),
                "--qy", str(qy),
                "--qz", str(qz),
                "--qw", str(qw),

                "--frame-id",
                "gripper_base",

                "--child-frame-id",
                "camera_color_optical_frame",
            ],

            output="screen",
        ),

        # ========================================================
        # 2. DaBai + MediaPipe
        #
        # 复用已经成功的视觉节点
        #
        # 它自己通过 pyorbbecsdk 打开 DaBai
        # 不需要 dabai.launch.py
        # ========================================================
        Node(
            package="piper_hand_follow",
            executable="hand_vision_node",
            name="hand_vision_node",

            output="screen",
        ),

        # ========================================================
        # 3. Hand stability
        # ========================================================
        Node(
            package="piper_handshake",
            executable="stability_detector",
            name="stability_detector",

            output="screen",
        ),
        # ========================================================
        # 4. Camera -> Base
        #
        # /locked_pose_camera
        #          ↓
        # /locked_pose_base
        # ========================================================
        Node(
            package="piper_handshake",
            executable="locked_pose_transformer",
            name="locked_pose_transformer",

            output="screen",
        ),

        # ========================================================
        # 5. Handshake target planner
        # ========================================================
        Node(
            package="piper_handshake",
            executable="handshake_planner",
            name="handshake_planner",
            parameters=[
                {
                    "approach_distance_m": 0.15,
                    "handshake_clearance_m": 0.05,
                }
            ],
            output="screen",
        ),
        # Node(
        #     package="piper_handshake",
        #     executable="handshake_controller",
        #     name="handshake_controller",
        #     parameters=[
        #         {
        #             # 第一次测试先 false
        #             "enable_motion": True,
        #             "workspace_x_min": 0.00,
        #             "workspace_x_max": 0.80,
        #             "workspace_y_min": -0.50,
        #             "workspace_y_max": 0.50,
        #             "workspace_z_min": 0.00,
        #             "workspace_z_max": 0.60,
        #             "max_move_distance_m": 0.45,
        #             "position_tolerance_m": 0.015,
        #             "motion_timeout_s": 15.0,
        #         }
        #     ],
        #     output="screen",
        # ),
        Node(
            package="piper_handshake",
            executable="moveit_auto_handshake_controller",
            name="moveit_auto_handshake_controller",

            parameters=[
                {
                    "planning_group": "arm",

                    "pose_link": "gripper_base",
                    "base_frame": "base_link",

                    "move_action": "/move_action",
                    "execute_action": "/execute_trajectory",

                    "joint_feedback_topic": "/feedback/joint_states",

                    # ======================================
                    # 总开关
                    # ======================================

                    "enable_auto_motion": True,

                    # 手稳定、Pose准备完成后自动开始
                    "auto_start_on_plan_ready": True,

                    # ======================================
                    # MoveIt planning
                    # ======================================

                    "planning_time_s": 5.0,
                    "planning_attempts": 5,

                    "position_tolerance_m": 0.010,
                    "orientation_tolerance_deg": 5.0,

                    # ======================================
                    # Approach
                    # ======================================

                    "approach_velocity_scaling": 0.10,
                    "approach_acceleration_scaling": 0.10,

                    # ======================================
                    # Handshake
                    # 更慢
                    # ======================================

                    "handshake_velocity_scaling": 0.05,
                    "handshake_acceleration_scaling": 0.05,

                    # ======================================
                    # Return HOME
                    # ======================================

                    "return_velocity_scaling": 0.10,
                    "return_acceleration_scaling": 0.10,

                    # ======================================
                    # 精确到位判断
                    # ======================================

                    "approach_verify_position_m": 0.020,
                    "approach_verify_orientation_deg": 10.0,

                    "handshake_verify_position_m": 0.015,
                    "handshake_verify_orientation_deg": 8.0,

                    "home_verify_joint_deg": 2.0,

                    "verify_stable_samples": 5,
                    "verify_timeout_s": 5.0,

                    # 到最终握手点停1.5秒
                    "handshake_dwell_s": 1.5,
                }
            ],

            output="screen",
        ),
    ])