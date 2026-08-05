"""Phase 6 -- the deployment, plus a browser window onto it.

    ros2 launch robot_rl_env monitor.launch.py policy:=runs/sac-seed0/policy.pt
    # then open http://localhost:8080

Includes ``deploy.launch.py`` unchanged -- same unpaused world, same policy
node, same wall-clock 20 Hz timer -- and adds two things: ``status_topic`` on
the policy node, and the monitor node beside it.

Not the launch to measure with
------------------------------
``make gap`` and ``deploy_eval.py`` use ``deploy.launch.py``, and should keep
using it. This one attaches a second node with its own subscriptions to the
same executor host and streams JSON to a browser at 20 Hz, on a container that
already only sustains a real-time factor of 0.3-0.5. That load is invisible in
the drawing and not invisible in the number.

Watch a run here; measure a run there.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "robot_rl_env"
STATUS_TOPIC = "/policy_status"


def generate_launch_description():
    share = get_package_share_directory(PKG)
    deploy_launch = os.path.join(share, "launch", "deploy.launch.py")

    args = [
        DeclareLaunchArgument(
            "policy",
            description="Path to a .pt written by robot_rl_env.export_policy. Required.",
        ),
        DeclareLaunchArgument(
            "host",
            default_value="0.0.0.0",
            description=(
                "The monitor binds this. 0.0.0.0 inside a container, because the "
                "browser is on the host and a container's loopback is its own. "
                "What keeps it private is the published port in compose, not this."
            ),
        ),
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument(
            "headless",
            default_value="true",
            description="No Gazebo GUI. The monitor is the view; that is the point.",
        ),
    ]

    deployment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(deploy_launch),
        launch_arguments={
            "policy": LaunchConfiguration("policy"),
            "headless": LaunchConfiguration("headless"),
            # The one thing this launch changes about the deployment itself.
            "status_topic": STATUS_TOPIC,
        }.items(),
    )

    monitor = Node(
        package=PKG,
        executable="monitor_node",
        name="monitor_node",
        output="screen",
        parameters=[
            {
                "host": LaunchConfiguration("host"),
                "port": LaunchConfiguration("port"),
                "status_topic": STATUS_TOPIC,
                "use_sim_time": False,  # as policy_node; see deploy.launch.py
            }
        ],
    )

    return LaunchDescription([*args, deployment, monitor])
