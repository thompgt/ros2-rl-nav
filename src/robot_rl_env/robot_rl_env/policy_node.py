"""Phase 4 -- the standalone ROS 2 inference node.

    ros2 run robot_rl_env policy_node --ros-args -p policy:=runs/sac-seed0/policy.pt

Runs against an **unpaused, real-time** world -- launch it with
``paused:=false``. Nothing here pauses, steps or teleports anything: this node
is what would run on a robot, and a robot has no world-control service.

What is deliberately *not* in this file
---------------------------------------
Any arithmetic. The observation assembly comes from ``observation.py``, the
action scaling from ``action.py``, and every decision from
``deploy.DeploymentController`` -- all three pure, all three shared with the
training env, all three tested on the host without a simulator. What is left
here is subscriptions, a timer, a publisher and parameter handling, which is
the part that genuinely needs ROS and the part that cannot be unit-tested
meaningfully anyway.

That split is not tidiness. Training/inference preprocessing divergence is the
classic silent failure of this project type: it produces no error, no log line
and no metric, just a policy that is quietly worse after deployment. The only
defence is that there is nothing here to diverge.

The contrast with the training loop, which is the measurement
-------------------------------------------------------------
``env.step`` publishes a velocity, advances the world by exactly 50 ms, and
*blocks* until the sensors carry a stamp at or beyond the new sim time. Every
observation is therefore exactly one action old, always.

This node cannot do any of that. It runs on a 20 Hz timer and reads whatever
arrived most recently, which is stale by a variable amount that depends on
bridge latency, executor scheduling and CPU load. Same policy, same
observation assembly, same episode rules -- different timing. The difference in
success rate between the two is the number Phase 4 exists to report, and
``deploy_eval.py`` measures it.

Goals
-----
A ``geometry_msgs/PoseStamped`` on ``/goal_pose`` (the topic RViz's "2D Goal
Pose" tool publishes, so a goal can be given by clicking). The goal is taken
**in the odom frame** and the header is checked, because a goal quietly
reinterpreted from another frame is wrong by however far odometry has drifted
-- small at the start of a run and growing, which presents as the policy
degrading over time rather than as a frame error.

No tf2 lookup, deliberately: a transform this node cannot verify is a
dependency on a tree that may not exist, and the failure would be silent in the
same way.
"""

from __future__ import annotations

import sys

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from robot_rl_env import contract
from robot_rl_env.deploy import DeploymentController, Outcome, TickStatistics
from robot_rl_env.export_policy import load_policy
from robot_rl_env.observation_node import ObservationAssembler

CMD_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
"""Depth 1, as in the training env: only the most recent command means
anything, and a queue of stale velocities is worse than a dropped one."""

GOAL_TOPIC = "/goal_pose"
ODOM_FRAME = "odom"

STALE_FOREVER = float("inf")
"""The age reported when no sample has ever arrived. The controller checks for
the missing sample before it looks at the age, so this is only ever a label --
but a zero here would read as "perfectly fresh" to anyone debugging."""


class PolicyNode(Node):
    """Drives ``/cmd_vel`` from an exported policy at ``contract.CONTROL_HZ``.

    Owns an :class:`ObservationAssembler` rather than subscribing itself, so
    the deployment path uses the same subscriber, the same QoS and the same
    pooling as training. It is a separate node and must be added to the same
    executor -- :func:`main` does that, and so does ``deploy_eval.py``.
    """

    def __init__(self, *, context=None, node_name: str = "policy_node"):
        super().__init__(node_name, context=context)

        self.declare_parameter("policy", "")
        self.declare_parameter("goal_topic", GOAL_TOPIC)
        self.declare_parameter("control_hz", float(contract.CONTROL_HZ))
        self.declare_parameter("goal_frame", ODOM_FRAME)

        policy_path = self.get_parameter("policy").value
        if not policy_path:
            raise ValueError(
                "the 'policy' parameter is required: pass "
                "--ros-args -p policy:=<path to a .pt written by "
                "robot_rl_env.export_policy>. There is no default, because a "
                "node that silently ran an untrained policy would look exactly "
                "like a policy that failed to learn."
            )
        self._goal_frame = self.get_parameter("goal_frame").value

        self.observations = ObservationAssembler(f"{node_name}_observations", context=context)
        self.controller = DeploymentController(load_policy(policy_path))
        # Accumulated across the whole session, not per episode: the thing worth
        # knowing is the distribution of observation staleness this deployment
        # ran under, and that is a property of the machine and the bridge rather
        # than of any one episode.
        self.stats = TickStatistics()

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", CMD_QOS)

        # The timer gets a MutuallyExclusiveCallbackGroup of its own: the
        # controller carries episode state across ticks and is not thread-safe,
        # and a MultiThreadedExecutor will happily run two timer callbacks at
        # once if the first is slow. Overlapping ticks would double-count the
        # step budget and interleave two policy evaluations against one
        # prev_action.
        control_hz = float(self.get_parameter("control_hz").value)
        self._tick_group = MutuallyExclusiveCallbackGroup()
        self._timer = self.create_timer(
            1.0 / control_hz, self._on_tick, callback_group=self._tick_group
        )

        self.create_subscription(
            PoseStamped,
            self.get_parameter("goal_topic").value,
            self._on_goal,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self._last_reason: str | None = None
        self.get_logger().info(
            f"policy={policy_path} control={control_hz:.1f} Hz "
            f"goal_topic={self.get_parameter('goal_topic').value} "
            f"goal_frame={self._goal_frame}; waiting for a goal"
        )

    # --- goals ----------------------------------------------------------------

    def _on_goal(self, msg: PoseStamped) -> None:
        frame = msg.header.frame_id
        # An empty frame_id is accepted: command-line publishers omit it, and
        # rejecting it would make the node awkward to drive by hand for no
        # safety gain. A *wrong* frame is rejected, because that is the case
        # that would otherwise be obeyed and be wrong.
        if frame and frame != self._goal_frame:
            self.get_logger().error(
                f"ignoring a goal in frame {frame!r}: this node takes goals in "
                f"{self._goal_frame!r} and does not transform them. A goal "
                f"reinterpreted from another frame is wrong by however far "
                f"odometry has drifted, and looks like the policy degrading."
            )
            return

        goal = (msg.pose.position.x, msg.pose.position.y)
        self.set_goal(goal)

    def set_goal(self, goal_xy: tuple[float, float]) -> None:
        """Start an episode toward ``goal_xy`` in the odom frame.

        Public and separate from the subscription callback so ``deploy_eval``
        can drive this node directly and measure the real deployment loop
        rather than a reimplementation of it.
        """
        self.controller.set_goal(goal_xy)
        self._last_reason = None
        self.get_logger().info(f"goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) in {self._goal_frame}")

    # --- the control loop -----------------------------------------------------

    def _on_tick(self) -> None:
        latest = self.observations.latest_sample()
        if latest is None:
            age = STALE_FOREVER
            command = self.controller.tick(
                pooled=None, robot_xy=None, robot_yaw=None, age=age
            )
        else:
            sample, age = latest
            command = self.controller.tick(
                pooled=sample.pooled, robot_xy=sample.xy, robot_yaw=sample.yaw, age=age
            )

        self._publish(command.linear, command.angular)
        self.stats.record(command.reason, age)
        self._log_transition(command)

    def _publish(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def _log_transition(self, command) -> None:
        """One line per change of reason, not one per tick.

        At 20 Hz an unconditional log is 1200 lines a minute and the one line
        that mattered -- the moment the watchdog first fired -- is unfindable
        in it.
        """
        if command.reason == self._last_reason:
            return
        self._last_reason = command.reason

        detail = ""
        if command.distance_to_goal is not None:
            detail += f" d={command.distance_to_goal:.2f}m"
        if command.min_lidar is not None:
            detail += f" clearance={command.min_lidar:.2f}m"

        line = f"{command.reason}{detail} step={command.step}"
        if command.outcome.is_terminal or command.reason in ("watchdog", "safety"):
            self.get_logger().warning(line)
        else:
            self.get_logger().info(line)

    def brake(self) -> None:
        """Publish a zero velocity. Called on the way out.

        A node that exits while the wheels are turning leaves a robot driving
        with nothing controlling it. The last message wins and there is nobody
        left to send another, so this is published rather than assumed.
        """
        self._publish(0.0, 0.0)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = PolicyNode()
        executor.add_node(node)
        executor.add_node(node.observations)
        node.get_logger().info(
            "spinning. The world must be running: launch it with paused:=false, "
            "or this node will report a watchdog stop forever -- a paused Gazebo "
            "publishes nothing at all."
        )
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 -- reported, then re-raised as an exit code
        print(f"policy_node: {error}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.brake()
            outcome = node.controller.outcome
            if outcome is not Outcome.IDLE:
                node.get_logger().info(f"final outcome: {outcome.value}")
            executor.shutdown(timeout_sec=2.0)
            node.observations.destroy_node()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
