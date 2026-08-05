"""Phase 4 -- the arena running free, with the policy node driving.

    ros2 launch robot_rl_env deploy.launch.py policy:=runs/sac-seed0/policy.pt

The counterpart of ``world.launch.py``'s training defaults: the world runs
**unpaused and in real time**, and nothing steps it. That is the whole point --
the policy meets sensor latency and timing jitter that the step-synchronized
training loop eliminated by construction, and the difference in success rate is
the Phase 4 measurement.

``use_sim_time`` is deliberately **false** for the policy node
------------------------------------------------------------
The bridges keep it, because their job is to stamp messages with the
simulator's clock. The node does not, because it is standing in for something
running on a robot, where the only clock is the wall clock.

The alternative -- driving the 20 Hz control timer from ``/clock`` -- looks
tempting, since it would hold the control period at exactly the 50 ms the
policy was trained on however slowly the simulator ran. It would also quietly
subtract part of the effect being measured: a real deployment's control loop
runs on a wall clock and does not get to slow down when the world does. The
same applies to the watchdog, which has to fire when the clock stops, and so
cannot be measured against that clock.

``deploy_eval.py`` reports the achieved real-time factor beside the results, so
a reader can tell whether the simulator was keeping up while the number was
taken.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "robot_rl_env"


def generate_launch_description():
    share = get_package_share_directory(PKG)
    world_launch = os.path.join(share, "launch", "world.launch.py")

    policy = LaunchConfiguration("policy")
    headless = LaunchConfiguration("headless")

    args = [
        DeclareLaunchArgument(
            "policy",
            description="Path to a .pt written by robot_rl_env.export_policy. Required.",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="true",
            description="No GUI process. Default true: this machine is headless.",
        ),
        DeclareLaunchArgument(
            "goal_topic",
            default_value="/goal_pose",
            description="PoseStamped goals, in the odom frame.",
        ),
    ]

    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(world_launch),
        # paused:=false is the one argument that matters here. The world must be
        # running: a paused Gazebo publishes nothing at all -- sensors update in
        # PostUpdate, which a paused world never runs -- so the node would sit
        # in a watchdog stop forever against a simulator that is working.
        launch_arguments={"paused": "false", "headless": headless}.items(),
    )

    policy_node = Node(
        package=PKG,
        executable="policy_node",
        name="policy_node",
        output="screen",
        parameters=[
            {
                "policy": policy,
                "goal_topic": LaunchConfiguration("goal_topic"),
                "use_sim_time": False,  # see the module docstring
            }
        ],
    )

    return LaunchDescription([*args, world, policy_node])
