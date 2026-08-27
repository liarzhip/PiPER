import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from moveit_configs_utils.launches import generate_warehouse_db_launch

from _moveit_config_builder import build_moveit_config, declare_common_args


def _launch(context):
    moveit_config = build_moveit_config(context)
    return list(generate_warehouse_db_launch(moveit_config).entities)


def generate_launch_description():
    return LaunchDescription(declare_common_args() + [OpaqueFunction(function=_launch)])
