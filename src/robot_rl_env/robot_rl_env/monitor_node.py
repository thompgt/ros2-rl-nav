"""Phase 6 -- the live web monitor's ROS end.

    ros2 launch robot_rl_env monitor.launch.py policy:=runs/sac-seed0/policy.pt
    # then open http://localhost:8080

A passive observer of a running deployment, plus one way to talk back: a click
becomes a ``PoseStamped`` on ``/goal_pose``, which is the same topic RViz's
"2D Goal Pose" tool publishes and the same one ``policy_node`` already listens
on. Nothing here pauses, steps or teleports anything -- for the same reason
``policy_node`` does not. The world it watches is the free-running one.

Why this is a separate node
---------------------------
It could have been a few extra publishers inside ``policy_node``. It is not,
because that node is the instrument the Phase 4 gap is measured with: an HTTP
server, a browser and 20 JSON frames a second sharing its process would put
the thing being measured and the thing measuring it on the same executor. Here,
the only trace the monitor leaves on the node under observation is one extra
publish per tick, which ``deploy_eval`` does not enable.

What it draws, and what it does not
-----------------------------------
It reads the scan through ``ObservationAssembler`` -- the same subscriber, QoS
and min-pooling the policy's observation comes from -- so the 20 sectors on
screen are the 20 numbers the policy received, not a prettier re-derivation
from ``/scan``. Everything else is in ``monitor.py``, and there is no geometry
in the browser at all.

It draws the arena in the world frame from a transform seeded at the spawn
pose, dead-reckoned from odometry. On a robot there is nothing better; here
there would be (Gazebo knows the true pose), and using it would hide exactly
the drift a deployed robot has to live with. So the drawn robot slides off the
drawn obstacles as odometry drifts, and that is the honest picture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from robot_rl_env import contract, monitor, monitor_server
from robot_rl_env.observation_node import ObservationAssembler
from robot_rl_env.policy_node import GOAL_TOPIC, ODOM_FRAME

STATUS_TOPIC = "/policy_status"
"""Where ``policy_node`` publishes its per-tick decision when asked to. The
monitor works without it -- it just cannot show the reason, the step count or
the outcome, because those live in the controller and nowhere on the bus."""


def installed_web_root():
    """The page, from the install share if there is one, else the source tree.

    Preferring the source tree would be wrong for a launched node; requiring
    the install would mean a colcon build to change a colour.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("robot_rl_env")) / "web"
        if (share / "index.html").is_file():
            return share
    except Exception:  # noqa: BLE001 -- not installed; the fallback is the point
        pass
    return monitor_server.WEB_ROOT


class MonitorNode(Node):
    """Serves the page, streams telemetry, forwards clicks as goals."""

    def __init__(self, *, context=None, node_name: str = "monitor_node"):
        super().__init__(node_name, context=context)

        self.declare_parameter("host", monitor_server.DEFAULT_HOST)
        self.declare_parameter("port", monitor_server.DEFAULT_PORT)
        self.declare_parameter("goal_topic", GOAL_TOPIC)
        self.declare_parameter("goal_frame", ODOM_FRAME)
        self.declare_parameter("status_topic", STATUS_TOPIC)
        self.declare_parameter("spawn", list(contract.ROBOT_SPAWN_POSE))

        spawn = tuple(float(v) for v in self.get_parameter("spawn").value)
        self._transform = monitor.spawn_transform(spawn)
        self._goal_frame = self.get_parameter("goal_frame").value

        self.observations = ObservationAssembler(f"{node_name}_observations", context=context)

        self._command: Twist | None = None
        self._status: dict | None = None
        self._goal_odom: tuple[float, float] | None = None

        group = ReentrantCallbackGroup()
        # Subscribing to /cmd_vel rather than asking the controller: what the
        # robot is actually being told is what reached the topic, and if a
        # publish is being dropped this is where it shows.
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10, callback_group=group)
        self.create_subscription(
            String,
            self.get_parameter("status_topic").value,
            self._on_status,
            10,
            callback_group=group,
        )
        # Goals from anywhere -- this page, RViz, or a command line -- so the
        # drawn goal is the one the node is actually pursuing.
        self.create_subscription(
            PoseStamped,
            self.get_parameter("goal_topic").value,
            self._on_goal_seen,
            10,
            callback_group=group,
        )
        self._goal_pub = self.create_publisher(
            PoseStamped, self.get_parameter("goal_topic").value, 10
        )

        self.server = monitor_server.MonitorServer(
            state=self.telemetry,
            on_goal=self.publish_goal,
            transform=self._transform,
            web_root=installed_web_root(),
            host=self.get_parameter("host").value,
            port=int(self.get_parameter("port").value),
            log=self.get_logger().debug,
        )
        self.server.start()
        self.get_logger().info(
            f"monitor on http://{self.get_parameter('host').value}:{self.server.port} "
            f"-- pose is dead-reckoned from spawn {spawn}, so drift off the drawn "
            f"obstacles is odometry error, not a drawing bug"
        )

    # --- what it hears ---------------------------------------------------------

    def _on_cmd(self, msg: Twist) -> None:
        self._command = msg

    def _on_status(self, msg: String) -> None:
        try:
            self._status = json.loads(msg.data)
        except json.JSONDecodeError:
            # Someone else publishing on this topic. Not fatal for a debug view.
            self.get_logger().warning(f"ignoring non-JSON status: {msg.data[:80]!r}")

    def _on_goal_seen(self, msg: PoseStamped) -> None:
        self._goal_odom = (msg.pose.position.x, msg.pose.position.y)

    # --- what it says ----------------------------------------------------------

    def telemetry(self) -> dict:
        """One frame, sampled now. Called from the server's stream thread."""
        latest = self.observations.latest_sample()
        sample, age = latest if latest is not None else (None, None)

        # The status carries the goal the controller is actually working on,
        # which beats the last one seen on the topic: a goal rejected for its
        # frame appears on the topic and is not being pursued.
        goal = self._goal_odom
        if self._status and self._status.get("goal"):
            goal = tuple(self._status["goal"])

        return monitor.telemetry_payload(
            transform=self._transform,
            pooled=sample.pooled if sample else None,
            robot_xy=sample.xy if sample else None,
            robot_yaw=sample.yaw if sample else None,
            goal_odom=goal,
            age=age,
            linear=self._command.linear.x if self._command else None,
            angular=self._command.angular.z if self._command else None,
            status=self._status,
        )

    def publish_goal(self, odom_xy, world_xy) -> None:
        """A click, as a ``PoseStamped`` in the frame the policy node accepts."""
        message = PoseStamped()
        message.header.frame_id = self._goal_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = float(odom_xy[0])
        message.pose.position.y = float(odom_xy[1])
        message.pose.orientation.w = 1.0  # goals are positions; yaw is unconstrained
        self._goal_pub.publish(message)
        self.get_logger().info(
            f"goal ({world_xy[0]:.2f}, {world_xy[1]:.2f}) world "
            f"-> ({odom_xy[0]:.2f}, {odom_xy[1]:.2f}) {self._goal_frame}"
        )

    def shutdown(self) -> None:
        self.server.stop()


def main(args=None) -> int:
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = MonitorNode()
        executor.add_node(node)
        executor.add_node(node.observations)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 -- reported, then an exit code
        print(f"monitor_node: {error}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.shutdown()
            executor.shutdown(timeout_sec=2.0)
            node.observations.destroy_node()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
