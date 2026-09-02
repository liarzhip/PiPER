from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'piper_pick_handover'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch'
            ),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),
            glob('config/*.yaml') +
            glob('config/*.json')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liar',
    maintainer_email='liarzhip@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scene_perception_node = piper_pick_handover.scene_perception_node:main',
            'target_transformer_node = piper_pick_handover.target_transformer_node:main',
            'target_lock_node = piper_pick_handover.target_lock_node:main',
            'table_scene_node = piper_pick_handover.table_scene_node:main',

            'grasp_planner_node = piper_pick_handover.grasp_planner_node:main',
            'observe_hand_planner_node = piper_pick_handover.observe_hand_planner_node:main',
            'handover_planner_node = piper_pick_handover.handover_planner_node:main',
            'work_home_controller_node = piper_pick_handover.work_home_controller_node:main',
            'moveit_executor_node = piper_pick_handover.moveit_executor_node:main',
            
            'manager_node = piper_pick_handover.manager_node:main',
        ],
    },
)
