"""Launch the arena world and the ROS <-> Gazebo bridge.

Defaults are chosen for training: headless, paused, sim time.

    ros2 launch robot_rl_env world.launch.py                   # headless, paused
    ros2 launch robot_rl_env world.launch.py headless:=false   # with the GUI
    ros2 launch robot_rl_env world.launch.py paused:=false     # free-running

``headless`` is the single switch; there is no separate ``gui`` argument,
because two arguments controlling one thing is how you end up with a "headless"
run that quietly spawns a GUI.

``paused:=true`` is the default because the whole architecture depends on the
env owning the advance of sim time -- see CLAUDE.md. Phase 4 deployment is the
only thing that should launch with ``paused:=false``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
)
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

PKG = "robot_rl_env"


def generate_launch_description():
    share = get_package_share_directory(PKG)
    world_file = os.path.join(share, "worlds", "arena.sdf")
    bridge_config = os.path.join(share, "config", "bridge.yaml")
    models_dir = os.path.join(share, "models")

    headless = LaunchConfiguration("headless")
    paused = LaunchConfiguration("paused")
    world = LaunchConfiguration("world")

    args = [
        DeclareLaunchArgument(
            "headless",
            default_value="true",
            description="No GUI process at all. Default true: this machine is headless.",
        ),
        DeclareLaunchArgument(
            "paused",
            default_value="true",
            description="Start paused. Required for step-synchronized training.",
        ),
        DeclareLaunchArgument(
            "world", default_value=world_file, description="Path to the SDF world."
        ),
    ]

    # gz must resolve model://diffbot out of the installed share directory.
    resource_path = SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models_dir)

    # `-s` is server-only: no GUI process at all, not a hidden window. `-r`
    # starts the sim running; omitting it leaves the world paused, which is the
    # default and what step-synchronized training requires. `-v 2` limits the
    # log to warnings and errors.
    server_cmd = PythonExpression(
        [
            "'gz sim -s -v 2 ' + ",
            "('-r ' if '", paused, "' == 'false' else '') + ",
            "'", world, "'",
        ]
    )

    gz_server = ExecuteProcess(
        cmd=[server_cmd],
        shell=True,
        output="screen",
        name="gz_server",
    )

    # Separate GUI process, only when explicitly asked for. It attaches to the
    # running server rather than starting a second simulation.
    gz_gui = ExecuteProcess(
        cmd=["gz sim -g -v 2"],
        shell=True,
        output="screen",
        name="gz_gui",
        condition=UnlessCondition(headless),
    )

    topic_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="topic_bridge",
        parameters=[{"config_file": bridge_config, "use_sim_time": True}],
        output="screen",
    )

    # Service bridge. These two services are what make step-synchronization
    # possible at all: without /world/arena/control the env cannot advance sim
    # time, and without /world/arena/set_pose it cannot randomize on reset.
    #
    # Both are provided by the gz-sim-user-commands-system plugin declared in
    # arena.sdf. If that plugin is ever dropped from the world, the services do
    # not exist, this bridge reports nothing, and every step() times out --
    # silence, not an error.
    service_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="service_bridge",
        arguments=[
            "/world/arena/control@ros_gz_interfaces/srv/ControlWorld",
            "/world/arena/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription(
        [*args, resource_path, gz_server, gz_gui, topic_bridge, service_bridge]
    )
