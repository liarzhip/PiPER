import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'piper_handshake'

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
            glob('config/*.json')+glob('config/*.yaml')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liar',
    maintainer_email='liarzhip@outlook.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stability_detector = piper_handshake.stability_detector:main',
            'locked_pose_transformer = piper_handshake.locked_pose_transformer:main',
            'handshake_planner = piper_handshake.handshake_planner:main',

            'moveit_auto_handshake_controller = piper_handshake.moveit_auto_handshake_controller:main',
        ],
    },
)
