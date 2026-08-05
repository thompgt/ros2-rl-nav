from glob import glob

from setuptools import find_packages, setup

package_name = "robot_rl_env"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/models/diffbot", glob("models/diffbot/*")),
        # The monitor page. Installed so a launched node finds it through the
        # ament index; monitor_node falls back to the source tree when there is
        # no install, so a colour can be changed without a colcon build.
        (f"share/{package_name}/web", glob("web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Thomas Pequegnot",
    maintainer_email="thomas.pequegnot04@gmail.com",
    description="Step-synchronized Gymnasium environment over ROS 2 / Gazebo Harmonic.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = robot_rl_env.policy_node:main",
            "monitor_node = robot_rl_env.monitor_node:main",
        ],
    },
)
