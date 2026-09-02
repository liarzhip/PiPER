import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def _load_handeye(json_path):
    fallback = {
        "x": -0.06778138216987459,
        "y": 0.00001623282691217559,
        "z": 0.036287740281927744,
        "qx": -0.12406555515595986,
        "qy": 0.12150446762495792,
        "qz": -0.6799007465212398,
        "qw": 0.7124460521687799,
    }

    def xyz(value):
        if isinstance(value, dict):
            return float(value["x"]), float(value["y"]), float(value["z"])
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return float(value[0]), float(value[1]), float(value[2])
        raise ValueError("unsupported translation/position format")

    def quat(value):
        if isinstance(value, dict):
            return (
                float(value["x"]),
                float(value["y"]),
                float(value["z"]),
                float(value["w"]),
            )
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return float(value[0]), float(value[1]), float(value[2]), float(value[3])
        raise ValueError("unsupported rotation/orientation format")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "position" in data and "orientation" in data:
            x, y, z = xyz(data["position"])
            qx, qy, qz, qw = quat(data["orientation"])
            return {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}

        if "translation" in data and "rotation" in data:
            x, y, z = xyz(data["translation"])
            qx, qy, qz, qw = quat(data["rotation"])
            return {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}

        keys = {"x", "y", "z", "qx", "qy", "qz", "qw"}
        if keys.issubset(data.keys()):
            return {
                "x": float(data["x"]),
                "y": float(data["y"]),
                "z": float(data["z"]),
                "qx": float(data["qx"]),
                "qy": float(data["qy"]),
                "qz": float(data["qz"]),
                "qw": float(data["qw"]),
            }

        raise ValueError("unrecognized hand-eye calibration JSON layout")

    except Exception as exc:
        print(f"[pick_handover_vision] Failed to parse handeye JSON: {exc}")

    print("[pick_handover_vision] Using validated fallback hand-eye calibration.")
    return fallback


def generate_launch_description():
    pkg_share = get_package_share_directory("piper_pick_handover")

    perception_yaml = os.path.join(pkg_share, "config", "perception.yaml")
    targets_yaml = os.path.join(pkg_share, "config", "targets.yaml")
    locking_yaml = os.path.join(pkg_share, "config", "locking.yaml")
    grasp_yaml = os.path.join(pkg_share, "config", "grasp.yaml")
    handeye_json = os.path.join(pkg_share, "config", "handeye_calibration.json")

    handeye = _load_handeye(handeye_json)

    handeye_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="handeye_static_tf",
        output="screen",
        arguments=[
            "--x", str(handeye["x"]),
            "--y", str(handeye["y"]),
            "--z", str(handeye["z"]),
            "--qx", str(handeye["qx"]),
            "--qy", str(handeye["qy"]),
            "--qz", str(handeye["qz"]),
            "--qw", str(handeye["qw"]),
            "--frame-id", "gripper_base",
            "--child-frame-id", "camera_color_optical_frame",
        ],
    )

    orbbec_lib = os.path.expanduser("~/PiPER_X/orbbec/pyorbbecsdk/install/lib")
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    display = os.environ.get("DISPLAY", ":0")
    xauthority = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))

    scene_perception = Node(
        package="piper_pick_handover",
        executable="scene_perception_node",
        name="scene_perception_node",
        output="screen",
        emulate_tty=True,
        additional_env={
            "PYTHONPATH": orbbec_lib + ":" + current_pythonpath,
            "DISPLAY": display,
            "XAUTHORITY": xauthority,
        },
        parameters=[
            perception_yaml,
            {"show_window": True},
        ],
    )

    target_transformer = Node(
        package="piper_pick_handover",
        executable="target_transformer_node",
        name="target_transformer_node",
        output="screen",
        parameters=[targets_yaml],
    )

    target_lock = Node(
        package="piper_pick_handover",
        executable="target_lock_node",
        name="target_lock_node",
        output="screen",
        parameters=[locking_yaml],
    )

    grasp_planner = Node(
        package="piper_pick_handover",
        executable="grasp_planner_node",
        name="grasp_planner_node",
        output="screen",
        parameters=[grasp_yaml],
    )

    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    "VISION ONLY: hand-eye TF + DaBai perception + target transform "
                    "+ staged target lock + grasp target generation."
                )
            ),
            LogInfo(
                msg=(
                    "Two-stage perception: first CASE only; after Lift + WORK_HOME, "
                    "manager arms a fresh Palm Final. No Palm Initial is used."
                )
            ),
            handeye_tf,
            scene_perception,
            target_transformer,
            target_lock,
            grasp_planner,
        ]
    )
