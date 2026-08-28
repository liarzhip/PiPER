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
        Node(
            package="piper_handshake",
            executable="handshake_controller",
            name="handshake_controller",
            parameters=[
                {
                    # 第一次测试先 false
                    "enable_motion": True,
                    "workspace_x_min": 0.00,
                    "workspace_x_max": 0.80,
                    "workspace_y_min": -0.50,
                    "workspace_y_max": 0.50,
                    "workspace_z_min": 0.00,
                    "workspace_z_max": 0.60,
                    "max_move_distance_m": 0.45,
                    "position_tolerance_m": 0.015,
                    "motion_timeout_s": 15.0,
                }
            ],
            output="screen",
        ),
    ])