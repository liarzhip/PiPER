import json
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory(
        "piper_hand_follow"
    )

    # -------------------------------------------------
    # 配置文件
    # -------------------------------------------------
    config_file = os.path.join(
        pkg_share,
        "config",
        "hand_follow.yaml"
    )

    calibration_file = os.path.join(
        pkg_share,
        "config",
        "handeye_calibration.json"
    )

    # -------------------------------------------------
    # 读取手眼标定结果
    # -------------------------------------------------
    with open(
        calibration_file,
        "r",
        encoding="utf-8"
    ) as f:
        calibration = json.load(f)
    x, y, z = calibration["position"]
    qx, qy, qz, qw = calibration["orientation"]

    # -------------------------------------------------
    # 是否允许真机运动
    # -------------------------------------------------
    enable_motion = LaunchConfiguration(
        "enable_motion"
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_motion",
            default_value="false",
            description=(
                "Enable real PIPER-X motion"
            ),
        ),

        # =================================================
        # Hand-eye static TF
        #
        # gripper_base
        #       ↓
        # camera_color_optical_frame
        # =================================================
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

        # =================================================
        # DaBai + MediaPipe
        # =================================================
        Node(
            package="piper_hand_follow",
            executable="hand_vision_node",
            name="hand_vision_node",

            parameters=[
                config_file
            ],
            output="screen",
        ),

        # =================================================
        # Visual servo controller
        # =================================================
        Node(
            package="piper_hand_follow",
            executable="hand_follow_controller",
            name="hand_follow_controller",
            parameters=[
                config_file,
                {
                    "enable_motion":
                        ParameterValue(
                            enable_motion,
                            value_type=bool
                        )
                },
            ],
            output="screen",
        ),
    ])