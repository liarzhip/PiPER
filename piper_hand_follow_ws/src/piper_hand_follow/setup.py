from glob import glob
import os
from setuptools import find_packages, setup

package_name = "piper_hand_follow"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")+ glob("config/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liar",
    maintainer_email="user@example.com",
    description="DaBai DC1 hand tracking and PIPER-X follow demo",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "hand_vision_node = piper_hand_follow.vision_node:main",
        "hand_follow_controller = piper_hand_follow.follow_controller_node:main",
    ]},
)
